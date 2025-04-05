import numpy as np
import math as ma
import random
from source.utils import *
from source.uncertainty_quantification import *

# TO DO: 'solutions' needs to be derived from the full integration CI and most robust UP method of the CIs
# Then `solutions_2` etc derived from other CIs and UP methods


class mediatedCausality():

    def __init__(self, plot_path, exam_name, name_list=['X','Z','Y','doing X','Y'], plot_flag=False, generate_flag=True, n_problems=360):

        if not is_divisible_by_9(n_problems):
            raise ValueError("\n The number of problems specified is not divisible by 9. Benchmark not created.")

        self.n_problems = n_problems # number of test problems
        self.plot_flag = plot_flag
        self.plot_path = plot_path
        self.exam_name = exam_name
        self.answer_proportions = [0.333,0.333,0.333] # ratios of A,B,C correct answers
        self.n_samples = 1000 # number of possible sample sizes for each generated causal scenario 
        self.min_power10_sample_size = 1 #2
        self.max_power10_sample_size = 6       
        self.difficulty_thresholds = np.array([0.1,0.25])         
        
        self.x_name = name_list[0] # 'smoke'
        self.z_name =  name_list[1] # 'have tar deposits in lungs'
        self.y_name =  name_list[2] # 'have lung cancer'
        self.x_name_verb =  name_list[3] # 'smoking'
        self.y_name_noun =  name_list[4] #'lung cancer'
   
        if generate_flag:
            self.make_problems() # all tests need this

    def make_problems(self): # all tests need this

        # 1) even split between easy, intermediate, and hard problems
        # 2) multiple UQ methods (t-distribution, Gaussian, bootstrap, full integration sample, full integration true)

        easy = {
            "n_problems": 0, # number of problems at this difficulty
            "n_A": 0, # number of problems at this difficulty with answer A
            "n_B": 0, # number of problems at this difficulty with answer B
            "n_C": 0, # number of problems at this difficulty with answer C    
            "n_samples": np.empty((), dtype=object), # total number of samples **in the given problem**                               
            "questions": np.empty((), dtype=object),
            "answers": np.empty((), dtype=object),           
            "P_Y1doX1": np.empty((), dtype=object),
            "P_Y1doX0": np.empty((), dtype=object),        
            "P_Y1doX1_CI": np.empty((), dtype=object),
            "P_Y1doX0_CI": np.empty((), dtype=object),
            "table": []                   
        }
        medm = easy.copy()
        hard = easy.copy()

        # binary variables
        xyz = np.array([[0,0,0],
                        [0,0,1],
                        [0,1,0],
                        [0,1,1],
                        [1,0,0],
                        [1,0,1],
                        [1,1,0],
                        [1,1,1]])

        j = 0
        while int(easy["n_problems"]+medm["n_problems"]+hard["n_problems"]) <  int(self.n_problems):  

            continue_flag = False
            
            # generate a causal scenario
            factor = np.logspace(self.min_power10_sample_size, self.max_power10_sample_size, num=self.n_samples,endpoint=True)
            generated_array = self.generate_dataset()
            dP = []
            diff_flag = 'none'

            P_Y1doX1_tmp = np.zeros([self.n_samples])
            P_Y1doX1_CI_tmp = np.zeros([self.n_samples])

            P_Y1doX0_tmp = np.zeros([self.n_samples])
            P_Y1doX0_CI_tmp = np.zeros([self.n_samples])            

            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)

            n_samples_tmp = np.zeros([self.n_samples])
            causality_tmp =  np.zeros([self.n_samples]) # (for plotting) 0 = uncertain, 1 = X causes Y, 2 = X does not cause Y
            table_tmp =  np.zeros([self.n_samples,8,4])
            
            for i in range(0,self.n_samples): 

                #samples = np.round(np.transpose(np.array([generated_array*factor[i]]))) # integers
                samples = np.transpose(np.array([generated_array*factor[i]]))  

                table = np.hstack((xyz,samples)) 
                table_tmp[i,:,:] = table

                P_Y1doX1_tmp[i],P_Y1doX1u,P_Y1doX1l,P_Y1doX0_tmp[i],P_Y1doX0u,P_Y1doX0l,n_samples_tmp[i] = self.causality_from_table( table )

                # Calculate dP (the difficulty)
                dP = np.abs(P_Y1doX1_tmp[i]-P_Y1doX0_tmp[i])
                if dP <= self.difficulty_thresholds[0]:
                    diff_flag = 'hard'
                elif dP <= self.difficulty_thresholds[1]:    
                    diff_flag = 'medm'  
                elif dP > self.difficulty_thresholds[1]:
                    diff_flag = 'easy'                      

                if diff_flag == 'hard' and hard["n_problems"] >= int(self.n_problems/3):
                    # if a hard problem and enough hard problems have been sampled already,
                    # then continue to another example
                    continue
                elif diff_flag == 'medm' and medm["n_problems"] >= int(self.n_problems/3):
                    # if a medium problem and enough medium problems have been sampled already,
                    # then continue to another example
                    continue
                elif diff_flag == 'easy' and easy["n_problems"] >= int(self.n_problems/3):
                    # if a easy problem and enough easy problems have been sampled already,
                    # then continue to another example
                    continue
                else:

                    if self.exam_name == 'mediatedCausalitySmoking':
                        questions_tmp[i] = (f"Please answer only with 'A', 'B', or 'C'. "
                                        f"The number of samples that do not {self.x_name}, do not {self.y_name}, and do not {self.z_name} is {int(table[0,3]):d}. "
                                        f"{int(table[1,3]):d} samples do not {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[2,3]):d} samples do not {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[3,3]):d} samples do not {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[4,3]):d} samples do {self.x_name}, do not {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[5,3]):d} samples do {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[6,3]):d} samples do {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[7,3]):d} samples do {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                        f"Does {self.x_name_verb} cause {self.y_name_noun}? If you quantitatively calculate uncertainty, use the 95% confidence level. Please only answer 'A' for yes, 'B' for no, or 'C' for uncertain, no other text."
                                        )
                    elif self.exam_name == 'mediatedCausalitySmokingWithMethod':
                        questions_tmp[i] = (f"Please answer only with 'A', 'B', or 'C'. "
                                        f"The number of samples that do not {self.x_name}, do not {self.y_name}, and do not {self.z_name} is {int(table[0,3]):d}. "
                                        f"{int(table[1,3]):d} samples do not {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[2,3]):d} samples do not {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[3,3]):d} samples do not {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[4,3]):d} samples do {self.x_name}, do not {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[5,3]):d} samples do {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                        f"{int(table[6,3]):d} samples do {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                        f"{int(table[7,3]):d} samples do {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                        f"Does {self.x_name_verb} cause {self.y_name_noun}? Use the front-door criterion to determine if smoking causes cancer from the provided data. Use the standard error of proportion and full range error propagation to calculate the most conservative estimate of the 95% confidence level for the front-door criterion calculation. Use the 95% confidence level intervals to answer 'A' for yes, 'B' for no, or 'C' for uncertain."
                                        )

 
                    P_Y1doX1_CI_tmp[i] = (P_Y1doX1u - P_Y1doX1l)/2.
                    P_Y1doX0_CI_tmp[i] = (P_Y1doX0u - P_Y1doX0l)/2.

                    # three outcomes (for plotting):
                    if P_Y1doX1l > P_Y1doX0u:
                        causality_tmp[i] = 1
                        answers_tmp[i] = 'A' # X causes Y
                    elif P_Y1doX0l > P_Y1doX1u:
                        causality_tmp[i] = 2
                        answers_tmp[i] = 'B' # not X causes Y
                    else:
                        causality_tmp[i] = 0
                        answers_tmp[i] = 'C' # Uncertain

            # Randomly choose a sample size for the given causal scenario
            random_choice_of_n_samples = np.random.randint(0, high=self.n_samples,size=self.n_samples)
            valid_idx = False 
            k=0 # loop over sample sizes   
            while not valid_idx:   

                select_idx = random_choice_of_n_samples[k]

                vars = {"P_Y1doX1_tmp": P_Y1doX1_tmp,
                        "P_Y1doX0_tmp": P_Y1doX0_tmp,
                        "P_Y1doX1_CI_tmp": P_Y1doX1_CI_tmp,
                        "P_Y1doX0_CI_tmp": P_Y1doX0_CI_tmp,
                        "answers_tmp": answers_tmp,
                        "questions_tmp": questions_tmp,
                        "n_samples_tmp": n_samples_tmp,
                        "table_tmp": table_tmp
                }        

                if diff_flag == 'hard':
                    valid_idx = self.update_dict(hard, vars, select_idx, valid_idx)
                elif diff_flag == 'medm':                  
                    valid_idx = self.update_dict(medm, vars, select_idx, valid_idx)
                elif diff_flag == 'easy':                  
                    valid_idx = self.update_dict(easy, vars, select_idx, valid_idx)

                k += 1 # loop over sample sizes 

                if k == int(self.n_samples):
                    # no necessary data available for this causal scenario
                    continue_flag = True # continue causal scenario while loop
                    valid_idx = True # break this while loop

            if continue_flag:
                # move on to the next causal scenario; generate another causal scenario
                continue

            # plot the chosen example
            if self.plot_flag: # make a plot of the 95% confidence interval
                create_missing_directory(self.plot_path)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                low_N = np.power(10.,self.min_power10_sample_size)
                high_N = np.power(10.,self.max_power10_sample_size)
                figname = self.plot_path + 'case_%i.png' %j
                fig = plt.figure(figsize=(12, 5))
                ax1=plt.subplot(1,2,1)
                P_Y1doX1u = P_Y1doX1_tmp + P_Y1doX1_CI_tmp/2.; P_Y1doX1l = P_Y1doX1_tmp - P_Y1doX1_CI_tmp/2.
                P_Y1doX0u = P_Y1doX0_tmp + P_Y1doX0_CI_tmp/2.; P_Y1doX0l = P_Y1doX0_tmp - P_Y1doX0_CI_tmp/2.
                plt.fill_between(n_samples_tmp, P_Y1doX1l, P_Y1doX1u, color='royalblue', alpha=0.2, label="95% CI P(Y=1|do(X=1))")
                plt.fill_between(n_samples_tmp, P_Y1doX0l, P_Y1doX0u, color='crimson', alpha=0.2, label="95% CI P(Y=1|do(X=0))")
                plt.plot(n_samples_tmp,P_Y1doX1_tmp,color='royalblue',linewidth=1)
                plt.plot(n_samples_tmp,P_Y1doX0_tmp,color='crimson',linewidth=1)
                plt.legend(loc=1,fontsize=13,framealpha=1.)
                plt.xlabel(r'$N_{samples}$',fontsize=18)
                plt.ylabel(r'Probability',fontsize=16)
                ax1.set_xscale("log")
                plt.axis([low_N,high_N,0.,1.])
                ax1=plt.subplot(1,2,2)
                plt.plot(n_samples_tmp,causality_tmp,color='black',linestyle='None',marker='o',markersize=10,linewidth=2)
                plt.plot(n_samples_tmp[select_idx],causality_tmp[select_idx],color='red',linestyle='None',marker='*',markersize=20,linewidth=2)
                plt.xlabel(r'$N_{samples}$',fontsize=18)
                ax1.set_xscale("log")
                plt.grid()
                plt.axis([low_N,high_N,-0.5,2.5])
                plt.title(diff_flag)
                plt.yticks([0.,1.,2.],[r'Uncertain',r'$X$ causes $Y$',r'$\neg X$ causes $Y$'],fontsize=14)
                plt.subplots_adjust(top=0.95, bottom=0.14, left=0.07, right=0.985, hspace=0.4, wspace=0.35)
                plt.savefig(figname,format="png"); plt.close(fig);            

            print('\n sum of easy, medium, hard problems = ',int(easy["n_problems"]+medm["n_problems"]+hard["n_problems"]))
            print(' easy, medium, hard problems = ',int(easy["n_problems"]),int(medm["n_problems"]),int(hard["n_problems"]))  
            print(' target total number of problems = ',  int(self.n_problems))       

            j += 1 # loop over causal examples

        # remove nones, add hard/medium/easy difficulty labels, combine all 
        questions = np.concatenate([easy["questions"][1:],medm["questions"][1:],hard["questions"][1:]])     
        answers = np.concatenate([easy["answers"][1:],medm["answers"][1:],hard["answers"][1:]])              
        P_Y1doX1 = np.concatenate([easy["P_Y1doX1"][1:],medm["P_Y1doX1"][1:],hard["P_Y1doX1"][1:]])
        P_Y1doX0 = np.concatenate([easy["P_Y1doX0"][1:],medm["P_Y1doX0"][1:],hard["P_Y1doX0"][1:]]) 
        P_Y1doX1_CI = np.concatenate([easy["P_Y1doX1_CI"][1:],medm["P_Y1doX1_CI"][1:],hard["P_Y1doX1_CI"][1:]])
        P_Y1doX0_CI = np.concatenate([easy["P_Y1doX0_CI"][1:],medm["P_Y1doX0_CI"][1:],hard["P_Y1doX0_CI"][1:]])       
        n_samples = np.concatenate([easy["n_samples"][1:],medm["n_samples"][1:],hard["n_samples"][1:]])    
        difficulty = np.empty(easy["n_problems"] + medm["n_problems"] + hard["n_problems"], dtype=object)
        difficulty[:easy["n_problems"]] = 'easy'
        difficulty[easy["n_problems"]:easy["n_problems"] + medm["n_problems"]] = 'intermediate'
        difficulty[-hard["n_problems"]:] = 'difficult'

        # now randomly shuffle
        idx = np.random.permutation(self.n_problems)

        self.questions = questions[idx]
        self.solutions  = answers[idx]
        self.difficulty  = difficulty[idx]
        self.n_samples  = n_samples[idx]
        self.P_Y1doX1  = P_Y1doX1[idx]
        self.P_Y1doX0  = P_Y1doX0[idx]
        self.P_Y1doX1_CI  = P_Y1doX1_CI[idx]
        self.P_Y1doX0_CI  = P_Y1doX0_CI[idx]  

        self.metadata = {
            "Name": self.exam_name,
            "P_Y1doX1": self.P_Y1doX1,
            "P_Y1doX0": self.P_Y1doX0,
            "P_Y1doX1_CI": self.P_Y1doX1_CI, # symmetric 
            "P_Y1doX0_CI": self.P_Y1doX0_CI,
            "n_samples": self.n_samples,
            "difficulty": self.difficulty,
            "A_count": np.count_nonzero(self.solutions == 'A'),
            "B_count": np.count_nonzero(self.solutions == 'B'),
            "C_count": np.count_nonzero(self.solutions == 'C'),
            "easy_count": np.count_nonzero(self.difficulty == 'easy'),
            "intermediate_count": np.count_nonzero(self.difficulty == 'intermediate'),
            "difficult_count": np.count_nonzero(self.difficulty == 'difficult'),            
            "n_problems": self.n_problems
        }

        print(' Done! ')

    def update_dict(self, info, vars, idx, valid_idx):

        if vars["answers_tmp"][idx] == 'A':
            if info["n_A"] < int(self.n_problems/9):
                info["n_A"] += 1 
                info["P_Y1doX1"] = np.append(info["P_Y1doX1"],vars["P_Y1doX1_tmp"][idx])
                info["P_Y1doX0"] = np.append(info["P_Y1doX0"],vars["P_Y1doX0_tmp"][idx])
                info["P_Y1doX1_CI"] = np.append(info["P_Y1doX1_CI"],vars["P_Y1doX1_CI_tmp"][idx])
                info["P_Y1doX0_CI"] = np.append(info["P_Y1doX0_CI"],vars["P_Y1doX0_CI_tmp"][idx])
                info["n_problems"] += 1     
                info["questions"] = np.append(info["questions"],vars["questions_tmp"][idx])              
                info["answers"] = np.append(info["answers"],vars["answers_tmp"][idx])  
                info["n_samples"] = np.append(info["n_samples"],vars["n_samples_tmp"][idx])  
                info["table"].append(vars["table_tmp"][idx,:,:])
                valid_idx = True       
            else:
                pass                

        elif vars["answers_tmp"][idx] == 'B':
            if info["n_B"] < int(self.n_problems/9):
                info["n_B"] += 1 
                info["P_Y1doX1"] = np.append(info["P_Y1doX1"],vars["P_Y1doX1_tmp"][idx])
                info["P_Y1doX0"] = np.append(info["P_Y1doX0"],vars["P_Y1doX0_tmp"][idx])
                info["P_Y1doX1_CI"] = np.append(info["P_Y1doX1_CI"],vars["P_Y1doX1_CI_tmp"][idx])
                info["P_Y1doX0_CI"] = np.append(info["P_Y1doX0_CI"],vars["P_Y1doX0_CI_tmp"][idx])   
                info["n_problems"] += 1     
                info["questions"] = np.append(info["questions"],vars["questions_tmp"][idx])              
                info["answers"] = np.append(info["answers"],vars["answers_tmp"][idx])  
                info["n_samples"] = np.append(info["n_samples"],vars["n_samples_tmp"][idx])   
                info["table"].append(vars["table_tmp"][idx,:,:])                
                valid_idx = True                       
            else:
                pass     

        elif vars["answers_tmp"][idx] == 'C':
            if info["n_C"] < int(self.n_problems/9):
                info["n_C"] += 1
                info["P_Y1doX1"] = np.append(info["P_Y1doX1"],vars["P_Y1doX1_tmp"][idx])
                info["P_Y1doX0"] = np.append(info["P_Y1doX0"],vars["P_Y1doX0_tmp"][idx])
                info["P_Y1doX1_CI"] = np.append(info["P_Y1doX1_CI"],vars["P_Y1doX1_CI_tmp"][idx])
                info["P_Y1doX0_CI"] = np.append(info["P_Y1doX0_CI"],vars["P_Y1doX0_CI_tmp"][idx])                   
                info["n_problems"] += 1     
                info["questions"] = np.append(info["questions"],vars["questions_tmp"][idx])              
                info["answers"] = np.append(info["answers"],vars["answers_tmp"][idx])  
                info["n_samples"] = np.append(info["n_samples"],vars["n_samples_tmp"][idx])    
                info["table"].append(vars["table_tmp"][idx,:,:])                
                valid_idx = True
            else:
                pass     
      
        return valid_idx


    def generate_dataset(self, size=8, sum_target_range=(0.7, 0.9), probability_of_pattern=0.90):

        chance = np.random.uniform(0, 1)

        if chance >= probability_of_pattern:
            array = np.random.uniform(0, 1, size)
            array = array / np.sum(array)
        else:
            two_sample_sum = np.random.uniform(sum_target_range[0], sum_target_range[1])
            two_samples = np.random.uniform(0.4, 0.6, 2)
            #two_samples = np.array([0.5,0.5])
            two_samples = two_samples / np.sum(two_samples) * two_sample_sum

            remaining_samples = np.random.uniform(0, 1, size-2)
            remaining_samples = remaining_samples / np.sum(remaining_samples) * (1.-two_sample_sum)
            array = np.append(two_samples,remaining_samples)
            np.random.shuffle(array)

        return array

    def verify_indices(self, answers, A_idx, B_idx, C_idx):
        if np.unique(answers[A_idx]) != 'A':
            print('\n ERROR: incorrect A answer indices')
        if np.unique(answers[B_idx]) != 'B':
            print('\n ERROR: incorrect B answer indices')
        if np.unique(answers[C_idx]) != 'C':
            print('\n ERROR: incorrect C answer indices')
        if int(len(A_idx)+len(B_idx)+len(C_idx)) != self.n_problems:
            print('\n ERROR: Number of problems not equivalent to n_problems')
            print(' problems made: ',int(len(A_idx)+len(B_idx)+len(C_idx)),'problems specified to made:',self.n_problems)
            #print(' Try increasing n_examples and/or n_sample_size')            
        return

    def get_indices(self, answers, answer_counts):
        indices = {'A': [], 'B': [], 'C': []}  # Dictionary to store all indices

        # Store all indices for 'A', 'B', and 'C'
        for i, value in enumerate(answers):
            if value in indices:
                indices[value].append(i)

        if len(indices['A']) >= answer_counts[0]:
            A_idx = random.sample(indices['A'],int(answer_counts[0])) #np.random.choice(len(indices['A']), answer_counts[0], replace=False)
        else:
            A_idx = indices['A']

        if len(indices['B']) >= answer_counts[1]:
            B_idx = random.sample(indices['B'],int(answer_counts[1])) #np.random.choice(len(indices['B']), answer_counts[1], replace=False)
        else:
            B_idx = indices['B']

        if len(indices['C']) >= answer_counts[2]:
            C_idx = random.sample(indices['C'],int(answer_counts[2])) #np.random.choice(len(indices['C']), answer_counts[2], replace=False)
        else:
            C_idx = indices['C']

        return A_idx,B_idx,C_idx

    def probability_x(self,arr,var_idx,outcome_idx):
        # P(x)
        # can change it to P_y or P_z by changing var_idx
        # var_idx = 0,1,2 for x,y,z
        # outcome_idx = 0 or 1
        filtered_rows = arr[arr[:, var_idx] == outcome_idx]
        return np.sum(filtered_rows[:, 3]) / np.sum(arr[:, 3]) # outputs P(x), for example

    def probability_z_given_x(self,data, x, z):
        # P(z|x)

        # Step 1: Get the values from the fourth column where rows match [x,0,z] and [x,1,z], then sum them
        mask1 = (data[:, 0] == x) & (data[:, 1] == 0) & (data[:, 2] == z)
        mask2 = (data[:, 0] == x) & (data[:, 1] == 1) & (data[:, 2] == z)
        sum_step1 = np.sum(data[mask1, 3]) + np.sum(data[mask2, 3])

        # Step 2: Get all values from the fourth column where the first column equals x, then sum them
        mask3 = (data[:, 0] == x)
        sum_step2 = np.sum(data[mask3, 3])

        # Step 3: Compute ratio (avoid division by zero)
        if sum_step2 != 0:
            return sum_step1 / sum_step2
        else:
            print('\n ERROR:  P_z_given_x == 0')
            return 1.0 #sum_step1 / 1e-16

    def probability_y_given_x_and_x(self,data, x, y, z):
        # P(y|x,z)
        mask1 = (data[:, 0] == x) & (data[:, 1] == y) & (data[:, 2] == z)
        numerator = data[mask1, 3]
        mask2 = (data[:, 0] == x) & (data[:, 1] == 0) & (data[:, 2] == z)
        mask3 = (data[:, 0] == x) & (data[:, 1] == 1) & (data[:, 2] == z)
        denominator = data[mask2, 3] + data[mask3, 3]
        return numerator / denominator

    def causality_from_table(self, data):
        # Calc P(x), P(z|x), P(Y=1|x,z) to get  P(Y=1|do(X=1))

        N = np.sum(data[:,3])

        # P(x)
        P_X0 = self.probability_x(data,0,0); P_X0u,P_X0l =get_95_CI(P_X0,N)
        P_X1 = self.probability_x(data,0,1); P_X1u,P_X1l =get_95_CI(P_X1,N)
        check_probability(P_X0)
        check_probability(P_X1)

        # P(z|x) = P(x,z)/P(x)
        P_Z0gX0 = self.probability_z_given_x(data, 0, 0);
        P_Z0gX0u,P_Z0gX0l =get_95_CI(P_Z0gX0,N)
        check_probability(P_Z0gX0)
        P_Z1gX0 = self.probability_z_given_x(data, 0, 1);
        P_Z1gX0u,P_Z1gX0l =get_95_CI(P_Z1gX0,N)
        check_probability(P_Z1gX0)
        P_Z0gX1 = self.probability_z_given_x(data, 1, 0);
        P_Z0gX1u,P_Z0gX1l =get_95_CI(P_Z0gX1,N)
        check_probability(P_Z0gX1)
        P_Z1gX1 = self.probability_z_given_x(data, 1, 1);
        check_probability(P_Z1gX1)
        P_Z1gX1u,P_Z1gX1l =get_95_CI(P_Z1gX1,N)

        # Get P(Y=1|x,z) = P(x,Y=1,z) / P(x,z)
        P_Y1gX0Z0 = self.probability_y_given_x_and_x(data, 0, 1, 0);
        P_Y1gX0Z0u,P_Y1gX0Z0l =get_95_CI(P_Y1gX0Z0,N)
        check_probability(P_Y1gX0Z0)
        P_Y1gX0Z1 = self.probability_y_given_x_and_x(data, 0, 1, 1);
        P_Y1gX0Z1u,P_Y1gX0Z1l =get_95_CI(P_Y1gX0Z1,N)
        check_probability(P_Y1gX0Z1)
        P_Y1gX1Z0 = self.probability_y_given_x_and_x(data, 1, 1, 0);
        P_Y1gX1Z0u,P_Y1gX1Z0l =get_95_CI(P_Y1gX1Z0,N)
        check_probability(P_Y1gX1Z0)
        P_Y1gX1Z1 = self.probability_y_given_x_and_x(data, 1, 1, 1);
        P_Y1gX1Z1u,P_Y1gX1Z1l =get_95_CI(P_Y1gX1Z1,N)
        check_probability(P_Y1gX1Z1)

        # compute P(Y=1|do(X=1))
        P_Y1doX1 = P_Z0gX1*(P_Y1gX0Z0*P_X0+P_Y1gX1Z0*P_X1) + P_Z1gX1*(P_Y1gX0Z1*P_X0+P_Y1gX1Z1*P_X1)
        check_probability(P_Y1doX1)

        # 95% confidence interval P(Y=1|do(X=1))
        P_Y1doX1u = P_Z0gX1u*(P_Y1gX0Z0u*P_X0u+P_Y1gX1Z0u*P_X1u) + P_Z1gX1u*(P_Y1gX0Z1u*P_X0u+P_Y1gX1Z1u*P_X1u)
        P_Y1doX1l = P_Z0gX1l*(P_Y1gX0Z0l*P_X0l+P_Y1gX1Z0l*P_X1l) + P_Z1gX1l*(P_Y1gX0Z1l*P_X0l+P_Y1gX1Z1l*P_X1l)

        # compute P(Y=1|do(X=0))
        P_Y1doX0 = P_Z0gX0*(P_Y1gX0Z0*P_X0+P_Y1gX1Z0*P_X1) + P_Z1gX0*(P_Y1gX0Z1*P_X0+P_Y1gX1Z1*P_X1)
        check_probability(P_Y1doX0)

        # 95% confidence interval P(Y=1|do(X=0))
        P_Y1doX0u = P_Z0gX0u*(P_Y1gX0Z0u*P_X0u+P_Y1gX1Z0u*P_X1u) + P_Z1gX0u*(P_Y1gX0Z1u*P_X0u+P_Y1gX1Z1u*P_X1u)
        P_Y1doX0l = P_Z0gX0l*(P_Y1gX0Z0l*P_X0l+P_Y1gX1Z0l*P_X1l) + P_Z1gX0l*(P_Y1gX0Z1l*P_X0l+P_Y1gX1Z1l*P_X1l)

        P_Y1doX1u = enforce_probability_bounds( P_Y1doX1u )
        P_Y1doX1l = enforce_probability_bounds( P_Y1doX1l )

        P_Y1doX0u = enforce_probability_bounds( P_Y1doX0u )
        P_Y1doX0l = enforce_probability_bounds( P_Y1doX0l )

        # needs to return a dict that includes all of the different UQ methods 
        # for each variable AND the different UP methods.

        return P_Y1doX1,P_Y1doX1u,P_Y1doX1l,P_Y1doX0,P_Y1doX0u,P_Y1doX0l,N


    def print_problems(self): # all tests need this
        for i in range(0,self.n_problems):
            print('\n')
            print(self.questions[i])
            print(self.solutions[i])

    def get_questions(self): # all tests need this
        return self.questions

    def get_solutions(self): # all tests need this
        return self.solutions

    def get_metadata(self): # all tests need this
        return self.metadata

    def get_difficulty(self): # all tests need this
        return self.difficulty

    def get_n_samples(self): # all tests need this
        return self.n_samples
 
#===============================================================================

# TEST

# x_name = 'smoke'
# z_name = 'have tar deposits in lungs'
# y_name = 'have lung cancer'
# x_name_verb = 'smoking'
# y_name_noun = 'lung cancer'

# exam_name = 'mediatedCausalitySmoking'
# plot_path = './figures/'
# exam = mediatedCausality(plot_path, exam_name,name_list=[x_name,z_name,y_name,x_name_verb,y_name_noun], plot_flag=True, n_problems=90)
