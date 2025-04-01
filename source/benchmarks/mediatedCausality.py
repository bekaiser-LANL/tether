import numpy as np
import math as ma
import random
import re
import os


class mediatedCausality():

    def __init__(self, plot_path, exam_name, name_list=['X','Z','Y','doing X','Y'], answer_proportions=[0.33,0.33,0.33], plot_flag=False, n_problems=100):

        self.n_problems = n_problems # number of test problems
        self.n_examples = 200 # number of examples that are sampled to generate test problems
        self.n_sample_sizes = 100 # number of samples sizes per example
        self.min_power10_sample_size = 2
        self.max_power10_sample_size = 6
        self.plot_flag = plot_flag
        self.plot_path = plot_path
        self.answer_proportions = answer_proportions
        self.exam_name = exam_name

        self.x_name = name_list[0] # 'smoke'
        self.z_name =  name_list[1] # 'have tar deposits in lungs'
        self.y_name =  name_list[2] # 'have lung cancer'
        self.x_name_verb =  name_list[3] # 'smoking'
        self.y_name_noun =  name_list[4] #'lung cancer'

        self.make_problems() # all tests need this


    def make_problems(self): # all tests need this
        # self.questions = [] # all tests need this
        # self.solutions = [] # all tests need this

        # N_test = self.n_problems
        # N_examples = self.n_examples
        # N_sample_sizes = self.n_sample_sizes
        # start_exp = self.min_power10_sample_size
        # stop_exp = self.max_power10_sample_size

        questions = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)
        answers = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)

        P_Y1doX1_ = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)
        P_Y1doX0_ = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)

        P_Y1doX1_CI = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)
        P_Y1doX0_CI = np.empty((self.n_examples, self.n_sample_sizes), dtype=object)

        # binary variables
        xyz = np.array([[0,0,0],
                        [0,0,1],
                        [0,1,0],
                        [0,1,1],
                        [1,0,0],
                        [1,0,1],
                        [1,1,0],
                        [1,1,1]])

        for j in range(0,self.n_examples):

            #print('\n Example ',j)
            factor = np.logspace(self.min_power10_sample_size, self.max_power10_sample_size, num=self.n_sample_sizes,endpoint=True)
            generated_array = self.generate_dataset()

            P_Y1doX1 = np.zeros([self.n_sample_sizes])
            P_Y1doX1u = np.zeros([self.n_sample_sizes])
            P_Y1doX1l = np.zeros([self.n_sample_sizes])

            P_Y1doX0 = np.zeros([self.n_sample_sizes])
            P_Y1doX0u = np.zeros([self.n_sample_sizes])
            P_Y1doX0l = np.zeros([self.n_sample_sizes])

            N_samples = np.zeros([self.n_sample_sizes])
            causality =  np.zeros([self.n_sample_sizes]) # 0 = uncertain, 1 = X causes Y, 2 = X does not cause Y

            for i in range(0,self.n_sample_sizes):

                samples = np.round(np.transpose(np.array([generated_array*factor[i]])))

                data = np.hstack((xyz,samples))

                P_Y1doX1[i],P_Y1doX1u[i],P_Y1doX1l[i],P_Y1doX0[i],P_Y1doX0u[i],P_Y1doX0l[i],N_samples[i] = self.causality_from_table( data )

                if self.exam_name == 'mediatedCausalitySmoking':
                    questions[j,i] = (f"Please answer only with 'A', 'B', or 'C'. "
                                      f"The number of samples that do not {self.x_name}, do not {self.y_name}, and do not {self.z_name} is {int(data[0,3]):d}. "
                                      f"{int(data[1,3]):d} samples do not {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[2,3]):d} samples do not {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[3,3]):d} samples do not {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[4,3]):d} samples do {self.x_name}, do not {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[5,3]):d} samples do {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[6,3]):d} samples do {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[7,3]):d} samples do {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                      f"Does {self.x_name_verb} cause {self.y_name_noun}? If you quantitatively calculate uncertainty, use the 95% confidence level. Please only answer 'A' for yes, 'B' for no, or 'C' for uncertain, no other text."
                                      )
                elif self.exam_name == 'mediatedCausalitySmokingWithMethod':
                    questions[j,i] = (f"Please answer only with 'A', 'B', or 'C'. "
                                      f"The number of samples that do not {self.x_name}, do not {self.y_name}, and do not {self.z_name} is {int(data[0,3]):d}. "
                                      f"{int(data[1,3]):d} samples do not {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[2,3]):d} samples do not {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[3,3]):d} samples do not {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[4,3]):d} samples do {self.x_name}, do not {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[5,3]):d} samples do {self.x_name}, do not {self.y_name}, and do {self.z_name}. "
                                      f"{int(data[6,3]):d} samples do {self.x_name}, do {self.y_name}, and do not {self.z_name}. "
                                      f"{int(data[7,3]):d} samples do {self.x_name}, do {self.y_name}, and do {self.z_name}. "
                                      f"Does {self.x_name_verb} cause {self.y_name_noun}? Use the front-door criterion to determine if smoking causes cancer from the provided data. Use the standard error of proportion and full range error propagation to calculate the most conservative estimate of the 95% confidence level for the front-door criterion calculation. Use the 95% confidence level intervals to answer 'A' for yes, 'B' for no, or 'C' for uncertain."
                                      )

                P_Y1doX1_[j,i] = P_Y1doX1[i]
                P_Y1doX1_CI[j,i] = (P_Y1doX1u[i] - P_Y1doX1l[i])/2.

                P_Y1doX0_[j,i] = P_Y1doX0[i]
                P_Y1doX0_CI[j,i] = (P_Y1doX0u[i] - P_Y1doX0l[i])/2.

                # three outcomes:
                if P_Y1doX1l[i] > P_Y1doX0u[i]:
                    causality[i] = 1
                    answers[j,i] = 'A' # X causes Y
                elif P_Y1doX0l[i] > P_Y1doX1u[i]:
                    causality[i] = 2
                    answers[j,i] = 'B' # not X causes Y
                else:
                    causality[i] = 0
                    answers[j,i] = 'C' # Uncertain

            if self.plot_flag: # make a plot of the 95% confidence interval
                self.create_missing_directory(self.plot_path)
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                low_N = np.power(10.,self.min_power10_sample_size)
                high_N = np.power(10.,self.max_power10_sample_size)
                figname = self.plot_path + 'case_%i.png' %j
                fig = plt.figure(figsize=(12, 5))
                ax1=plt.subplot(1,2,1)
                plt.fill_between(N_samples, P_Y1doX1l, P_Y1doX1u, color='royalblue', alpha=0.2, label="95% CI P(Y=1|do(X=1))")
                plt.fill_between(N_samples, P_Y1doX0l, P_Y1doX0u, color='crimson', alpha=0.2, label="95% CI P(Y=1|do(X=0))")
                plt.plot(N_samples,P_Y1doX1,color='royalblue',linewidth=1)
                plt.plot(N_samples,P_Y1doX0,color='crimson',linewidth=1)
                plt.legend(loc=1,fontsize=13,framealpha=1.)
                plt.xlabel(r'$N_{samples}$',fontsize=18)
                plt.ylabel(r'Probability',fontsize=16)
                ax1.set_xscale("log")
                plt.axis([low_N,high_N,0.,1.])
                ax1=plt.subplot(1,2,2)
                plt.plot(N_samples,causality,color='black',linestyle='None',marker='o',markersize=10,linewidth=2)
                plt.xlabel(r'$N_{samples}$',fontsize=18)
                ax1.set_xscale("log")
                plt.grid()
                plt.axis([low_N,high_N,-0.5,2.5])
                plt.yticks([0.,1.,2.],[r'Uncertain',r'$X$ causes $Y$',r'$\neg X$ causes $Y$'],fontsize=14)
                plt.subplots_adjust(top=0.95, bottom=0.14, left=0.07, right=0.985, hspace=0.4, wspace=0.35)
                plt.savefig(figname,format="png"); plt.close(fig);


        P_Y1doX1 = P_Y1doX1_.flatten()
        P_Y1doX1_CI = P_Y1doX1_CI.flatten()

        P_Y1doX0 = P_Y1doX0_.flatten()
        P_Y1doX0_CI = P_Y1doX0_CI.flatten()

        questions = questions.flatten()
        answers = answers.flatten()

        # get the count of each type of correct answer ('A', 'B', or 'C')
        # results in: [N_A,N_B,N_C]
        #answer_counts = [round(x * self.n_problems) for x in self.answer_proportions]
        answer_counts = [int(self.n_problems/(int(1/self.answer_proportions[0]))),
                         int(self.n_problems/(int(1/self.answer_proportions[1]))),
                         int(self.n_problems/(int(1/self.answer_proportions[2])))]
        print('\n Number of correct A,B,C answers (should sum to n_problems) = ',answer_counts)
        #print(' np.sum(answer_counts) = ',np.sum(answer_counts))
        print(' MAKE SURE n_problems IS DIVISIBLE BY 3!')


        # get randomly chosen 'A', 'B', or 'C' problem indices that when counted
        # add up to the correct proportions:
        A_idx,B_idx,C_idx = self.get_indices( answers, answer_counts)
        self.verify_indices(answers, A_idx, B_idx, C_idx)
        idx = np.concatenate((A_idx,B_idx,C_idx),dtype=int)
        np.random.shuffle(idx)
        

        # # a sanity check
        #answers2 = answers[idx]
        # A_idx = self.get_str_indices(answers2,'A')
        # N_A = len(A_idx)
        # B_idx = self.get_str_indices(answers2,'B')
        # N_B = len(B_idx)
        # C_idx = self.get_str_indices(answers2,'C')
        # N_C = len(C_idx)
        # print(' len(A_idx) = ',len(A_idx))
        # print(' len(B_idx) = ',len(B_idx))
        # print(' len(C_idx) = ',len(C_idx))
        # self.verify_indices(answers2, A_idx, B_idx, C_idx)

        self.questions = questions[idx]
        self.solutions  = answers[idx]

        # print(' len(self.questions) = ',len(self.questions))
        # print(' len(self.solutions) = ',len(self.solutions))

        self.solutions_P_Y1doX1    = P_Y1doX1[idx]
        self.solutions_P_Y1doX0    = P_Y1doX0[idx]
        self.solutions_dP          = np.abs(P_Y1doX1[idx]-P_Y1doX0[idx])
        self.solutions_P_Y1doX1_CI = P_Y1doX1_CI[idx]
        self.solutions_P_Y1doX0_CI = P_Y1doX0_CI[idx]

        self.solutions_A_count = np.sum(self.solutions == 'A')
        self.solutions_B_count = np.sum(self.solutions == 'B')
        self.solutions_C_count = np.sum(self.solutions == 'C')

        #print(' self.solutions_A_count = ',self.solutions_A_count)
        #print(' self.solutions_B_count = ',self.solutions_B_count)        
        #print(' self.solutions_C_count = ',self.solutions_C_count)        

        self.metadata = {
            "Name": self.exam_name,
            "dP": self.solutions_dP,
            "P_Y1doX1": self.solutions_P_Y1doX1,
            "P_Y1doX0": self.solutions_P_Y1doX0,
            "P_Y1doX1_CI": self.solutions_P_Y1doX1_CI,
            "P_Y1doX0_CI": self.solutions_P_Y1doX0_CI,
            "A_count": self.solutions_A_count,
            "B_count": self.solutions_B_count,
            "C_count": self.solutions_C_count,
            "n_problems": self.n_problems
        }

    def check_probability(self,P):
        if P > 1.:
            print('\n ERROR: Probability > 1')
        elif P < 0.:
            print('\n ERROR: Probability < 0')
        return

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
            print(' Try increasing n_examples and/or n_sample_size')            
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


    def create_missing_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def standard_error(self,P,N):
        # standard error for proportion
        return np.sqrt((P*(1.-P))/N) # <--- Need a reference for this!

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


    def get_95_CI(self,P,N):
        se = self.standard_error(P,N)
        return P+1.96*se,P-1.96*se

    def probability_y_given_x_and_x(self,data, x, y, z):
        # P(y|x,z)
        mask1 = (data[:, 0] == x) & (data[:, 1] == y) & (data[:, 2] == z)
        numerator = data[mask1, 3]
        mask2 = (data[:, 0] == x) & (data[:, 1] == 0) & (data[:, 2] == z)
        mask3 = (data[:, 0] == x) & (data[:, 1] == 1) & (data[:, 2] == z)
        denominator = data[mask2, 3] + data[mask3, 3]
        return numerator / denominator

    def get_str_indices(self,lst,str):
        return [index for index, value in enumerate(lst) if value == str]

    def enforce_probability_bounds(self, var ):
        if var > 1.:
            var = 1.
        elif var < 0.:
            var = 0.
        return var

    def causality_from_table(self, data ):
        # Calc P(x), P(z|x), P(Y=1|x,z) to get  P(Y=1|do(X=1))

        N = np.sum(data[:,3])

        # P(x)
        P_X0 = self.probability_x(data,0,0); P_X0u,P_X0l = self.get_95_CI(P_X0,N)
        P_X1 = self.probability_x(data,0,1); P_X1u,P_X1l = self.get_95_CI(P_X1,N)
        self.check_probability(P_X0)
        self.check_probability(P_X1)

        # P(z|x) = P(x,z)/P(x)
        P_Z0gX0 = self.probability_z_given_x(data, 0, 0);
        P_Z0gX0u,P_Z0gX0l = self.get_95_CI(P_Z0gX0,N)
        self.check_probability(P_Z0gX0)
        P_Z1gX0 = self.probability_z_given_x(data, 0, 1);
        P_Z1gX0u,P_Z1gX0l = self.get_95_CI(P_Z1gX0,N)
        self.check_probability(P_Z1gX0)
        P_Z0gX1 = self.probability_z_given_x(data, 1, 0);
        P_Z0gX1u,P_Z0gX1l = self.get_95_CI(P_Z0gX1,N)
        self.check_probability(P_Z0gX1)
        P_Z1gX1 = self.probability_z_given_x(data, 1, 1);
        self.check_probability(P_Z1gX1)
        P_Z1gX1u,P_Z1gX1l = self.get_95_CI(P_Z1gX1,N)

        # Get P(Y=1|x,z) = P(x,Y=1,z) / P(x,z)
        P_Y1gX0Z0 = self.probability_y_given_x_and_x(data, 0, 1, 0);
        P_Y1gX0Z0u,P_Y1gX0Z0l = self.get_95_CI(P_Y1gX0Z0,N)
        self.check_probability(P_Y1gX0Z0)
        P_Y1gX0Z1 = self.probability_y_given_x_and_x(data, 0, 1, 1);
        P_Y1gX0Z1u,P_Y1gX0Z1l = self.get_95_CI(P_Y1gX0Z1,N)
        self.check_probability(P_Y1gX0Z1)
        P_Y1gX1Z0 = self.probability_y_given_x_and_x(data, 1, 1, 0);
        P_Y1gX1Z0u,P_Y1gX1Z0l = self.get_95_CI(P_Y1gX1Z0,N)
        self.check_probability(P_Y1gX1Z0)
        P_Y1gX1Z1 = self.probability_y_given_x_and_x(data, 1, 1, 1);
        P_Y1gX1Z1u,P_Y1gX1Z1l = self.get_95_CI(P_Y1gX1Z1,N)
        self.check_probability(P_Y1gX1Z1)

        # compute P(Y=1|do(X=1))
        P_Y1doX1 = P_Z0gX1*(P_Y1gX0Z0*P_X0+P_Y1gX1Z0*P_X1) + P_Z1gX1*(P_Y1gX0Z1*P_X0+P_Y1gX1Z1*P_X1)
        self.check_probability(P_Y1doX1)

        # 95% confidence interval P(Y=1|do(X=1))
        P_Y1doX1u = P_Z0gX1u*(P_Y1gX0Z0u*P_X0u+P_Y1gX1Z0u*P_X1u) + P_Z1gX1u*(P_Y1gX0Z1u*P_X0u+P_Y1gX1Z1u*P_X1u)
        P_Y1doX1l = P_Z0gX1l*(P_Y1gX0Z0l*P_X0l+P_Y1gX1Z0l*P_X1l) + P_Z1gX1l*(P_Y1gX0Z1l*P_X0l+P_Y1gX1Z1l*P_X1l)

        # compute P(Y=1|do(X=0))
        P_Y1doX0 = P_Z0gX0*(P_Y1gX0Z0*P_X0+P_Y1gX1Z0*P_X1) + P_Z1gX0*(P_Y1gX0Z1*P_X0+P_Y1gX1Z1*P_X1)
        self.check_probability(P_Y1doX0)

        # 95% confidence interval P(Y=1|do(X=0))
        P_Y1doX0u = P_Z0gX0u*(P_Y1gX0Z0u*P_X0u+P_Y1gX1Z0u*P_X1u) + P_Z1gX0u*(P_Y1gX0Z1u*P_X0u+P_Y1gX1Z1u*P_X1u)
        P_Y1doX0l = P_Z0gX0l*(P_Y1gX0Z0l*P_X0l+P_Y1gX1Z0l*P_X1l) + P_Z1gX0l*(P_Y1gX0Z1l*P_X0l+P_Y1gX1Z1l*P_X1l)

        P_Y1doX1u = self.enforce_probability_bounds( P_Y1doX1u )
        P_Y1doX1l = self.enforce_probability_bounds( P_Y1doX1l )

        P_Y1doX0u = self.enforce_probability_bounds( P_Y1doX0u )
        P_Y1doX0l = self.enforce_probability_bounds( P_Y1doX0l )

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

#===============================================================================

# TEST

# x_name = 'smoke'
# z_name = 'have tar deposits in lungs'
# y_name = 'have lung cancer'
# x_name_verb = 'smoking'
# y_name_noun = 'lung cancer'
#
# exam = mediatedCausality(name_list=[x_name,z_name,y_name,x_name_verb,y_name_noun], plot_flag=True)
#
# with open("./exam.txt", "w") as file:
#     # Write the header
#     file.write("Mediated Causality Exam Questions and Answers\n")
#     file.write("=" * 40 + "\n")
#     file.write("This file contains a series of exam questions along with their correct answers and probabilistic estimates.\n")
#     file.write("Generated answers:\n")
#     file.write("- %.2f percent X causes Y \n" %(exam.solutions_A_count/len(exam.solutions)))
#     file.write("- %.2f percent not X causes Y \n" %(exam.solutions_B_count/len(exam.solutions)))
#     file.write("- %.2f percent uncertain \n" %(exam.solutions_C_count/len(exam.solutions)))
#     file.write("- %i total questions \n" %(len(exam.solutions)))
#     file.write("=" * 40 + "\n\n")
#
#     # Write exam questions and results
#     for i in range(len(exam.solutions)):
#         file.write('\n\nProblem %i' % (i + 1))
#         file.write('\n' + exam.questions[i])
#         file.write('\nCorrect answer: ' + exam.solutions[i])
#         file.write('\nLLM response:')
#         file.write('\nP(Y=1|do(X=1)) = %.3f +/- %.3f' % (exam.solutions_P_Y1doX1[i], exam.solutions_P_Y1doX1_CI[i]))
#         file.write('\nP(Y=1|do(X=0)) = %.3f +/- %.3f' % (exam.solutions_P_Y1doX0[i], exam.solutions_P_Y1doX0_CI[i]))
