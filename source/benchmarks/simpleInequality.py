import numpy as np
import math as ma
import random
import sys
from scipy.stats import t
from .mediated_causality import MediatedCausality
from source.utils import QuestionBank
from source.utils import create_missing_directory
from source.utils import is_divisible_by_9
from source.utils import check_probability

class simpleInequality():


    def __init__(self, exam_name, n_numbers = 100, **kwargs):

        #self.plot_path = plot_path
        self.exam_name = exam_name

        #generation parameters:
        self.plot_flag = kwargs.get('plot_flag', False)
        self.generate_flag = kwargs.get('generate_flag', True)
        self.verbose = kwargs.get('verbose', False)
        self.n_numbers = n_numbers #length of each vector
        self.answer_proportions = kwargs.get(
            "answer_proportions",
            [0.333, 0.333, 0.333], # Ratios of A, B, C correct answers
        )
        self.n_problems = kwargs.get('n_problems', 18)
        self.n_samples = kwargs.get('n_samples', self.n_problems/len(self.answer_proportions))
        self.n_samples = int(self.n_samples)
        self.difficulty_thresholds = kwargs.get(
            'difficulty_thresholds',
            np.array([0.66,1.33])
        )
        #self.ci_method = (exam_name).split('_')[1]
        #self.exam_name_wo_ci_method = (exam_name).split('_')[0]
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        if not is_divisible_by_9(self.n_problems):
            raise ValueError(
                "\n The number of problems specified is not divisible by 9."
                "Benchmark not created."
            )
        if self.generate_flag: 
            self.make_problems()

    def get_prompts(self):
        """ Get questions for different kinds of inequality tests """
        v1, v2 = self.generate_dataset()

        # Convert the list of numbers to a space-separated string
        v1numbers_str = " ".join(map(str, v1)) 
        v2numbers_str = " ".join(map(str, v2)) 
        q = []
        q = f"Vector 1: {v1numbers_str} Vector 2: {v2numbers_str} Which vector has the higher mean? A: Vector 1 B: Vector 2 C: Uncertain Answer with one letter only: A, B, or C. Answer:"
        return v1, v2, q

    def make_plot(self,problem):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            create_missing_directory(self.plot_path)
            import matplotlib # pylint: disable=import-outside-toplevel
            matplotlib.use('Agg') # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel
            low_n = np.power(10.,self.min_power10_sample_size)
            high_n = np.power(10.,self.max_power10_sample_size)
            figname = f"{self.plot_path}{self.ci_method}_example_{problem['example_idx']}.png"
            fig = plt.figure(figsize=(12, 5))
            ax1=plt.subplot(1,2,1)
            plt.fill_between(
                problem["n_samples_all"],
                problem["p_diff_ci_lower_all"],
                problem["p_diff_ci_upper_all"],
                color="royalblue",
                alpha=0.2,
                label="95% CI"
            )
            plt.plot(problem["n_samples_all"],problem["p_diff_all"],color='royalblue',linewidth=1)
            plt.plot(
                problem["n_samples"],
                problem["p_diff"],
                color='royalblue',
                linestyle='None',
                marker='*',
                markersize=20,
                linewidth=2
            )
            plt.legend(loc=1,fontsize=13,framealpha=1.)
            plt.xlabel(r'$N_{samples}$',fontsize=18)
            plt.ylabel(r'Probability',fontsize=16)
            ax1.set_xscale("log")
            plt.axis([low_n,high_n,-1.,1.])
            ax1=plt.subplot(1,2,2)
            plt.plot(
                problem["n_samples_all"],
                problem["causality_all"],
                color='black',
                linestyle='None',
                marker='o',
                markersize=10,
                linewidth=2
            )
            plt.plot(
                problem["n_samples"],
                problem["causality"],
                color='red',
                linestyle='None',
                marker='*',
                markersize=20,
                linewidth=2
            )
            plt.xlabel(r'$N_{samples}$',fontsize=18)
            ax1.set_xscale("log")
            plt.grid()
            plt.axis([low_n,high_n,-0.5,2.5])
            plt.title(problem["difficulty"])
            plt.yticks(
                [0.,1.,2.],
                [r'Uncertain (C)',r'$\neg X$ causes $Y$ (B)',r'$X$ causes $Y$ (A)'],
                fontsize=14
            )
            plt.subplots_adjust(
                top=0.95,
                bottom=0.14,
                left=0.07,
                right=0.985,
                hspace=0.4,
                wspace=0.35
            )
            plt.savefig(figname,format="png")
            plt.close(fig)


    def make_problems(self): 
        """ Generate simple Inequality questions for the LLMs """

        qb = QuestionBank(target_per_bin =int(self.n_problems/9))
        test_complete = False
        example_idx = 0
        while not test_complete:
            
            #mean1, mean2, diff = self.find_mean_difference(v1,v2)
            # these range over varied n_samples:
            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)
            difficulty_tmp = np.empty(self.n_samples, dtype=object)
            n_samples_tmp = np.zeros([self.n_samples])
            mean_diff_tmp = np.zeros([self.n_samples])
            for i in reversed(range(self.n_samples)):


                #get questions:
                questions_tmp[i] = self.get_prompts()[2]
               # print(questions_tmp[i])
                #calculate the difficulty level
                difficulty_tmp[i] = self.assign_difficulty(self.get_prompts()[0],self.get_prompts()[1])
                #record the solutions:
                answers_tmp[i] = self.record_solutions(self.get_prompts()[0],self.get_prompts()[1])[1]
                #print(answers_tmp[i])
                mean_diff_tmp[i] = self.find_mean_difference(self.get_prompts()[0],self.get_prompts()[1])[2]
        
            # Randomly select one case from the generated causal examples
            # with different numbers of samples:
            random_choice_of_n_samples = np.random.randint(
                0,
                high=self.n_samples,
                size=self.n_samples
            )

            # Make sure the random choice has a non-NaN p_diff:
            mean_diff_is_not_nan = False
            k = 0
            subsample_idx = 0
            while not mean_diff_is_not_nan:
                subsample_idx = random_choice_of_n_samples[k]
                if np.isnan(mean_diff_tmp[subsample_idx]):
                    k += 1
                else:
                    mean_diff_is_not_nan = True

            problem = {
                "question": questions_tmp[subsample_idx],
                "solution": answers_tmp[subsample_idx],
                "difficulty": difficulty_tmp[subsample_idx],
                "n_samples": n_samples_tmp[subsample_idx],
                "n_samples_all": n_samples_tmp,
                "subsample_idx": subsample_idx,
                "example_idx": example_idx,
                "name": self.exam_name
            }

            if qb.add_question(
                problem["question"],
                problem["solution"],
                problem["difficulty"],
                metadata={k: v for k, v in problem.items() if k not in {"question", "solution", "difficulty"}}
                ):
                #continue
                self.make_plot(problem)

            # Check if ready:
            if qb.is_full():
                final_set = qb.get_balanced_set()
                if self.verbose:
                    print("Test is complete:", len(final_set), "questions")
                test_complete = True
                #Pull attributes from qb
                qb.n_samples = np.array([q['metadata']['n_samples'] for q in qb.get_balanced_set()])
                qb.name = np.array([q['metadata']['name'] for q in qb.get_balanced_set()])
                qb.example_idx = np.array([q['metadata']['example_idx'] for q in qb.get_balanced_set()])
                qb.solution = [q['solution'] for q in qb.get_balanced_set()]
                qb.question = [q['question'] for q in qb.get_balanced_set()]
                qb.difficulty = [q['difficulty'] for q in qb.get_balanced_set()]
                for name, value in qb.__dict__.items():
                    setattr(self, name, value)
            else:
                if self.verbose:
                    print("Still building test. Current count:", qb.count())
                example_idx += 1 # loop over examples 
        print(' Done! ')


    #def make_problems(self): # all tests need this
    #    self.questions = [] # all tests need this
    #    self.solutions = [] # all tests need this
    #    for i in range(0,self.n_problems): # all tests need this
#
#            v1, v2, q_str = self.get_prompts()
#            self.questions = np.append(self.questions,q_str)
#
#            #ans_str1 = mean#vector_str = "[" + ", ".join('{:.2f}'.format(x) for x in v1) + "]"
#            #ans_str2 = mean#vector_str = "[" + ", ".join('{:.2f}'.format(x) for x in v2) + "]"
#            label = self.find_mean_difference(v1,v2)
#            #print('label=',label)
#            self.solutions = np.append(self.solutions,label)

    def generate_vector(self, target_mean, target_std, length):
        length = self.n_numbers
        vec = np.random.randn(length)
        vec -= np.mean(vec)
        vec /= np.std(vec)
        vec *= target_std
        vec += target_mean
        vec = np.round(vec, 2) #round to 2 decimal places
        return vec

    def generate_vector_pair(self, mean_diff_range, length):
        length = self.n_numbers
        while True:
            mean1 = np.random.uniform(-1, 1)
            diff = np.random.uniform(*mean_diff_range)
            # Randomly decide which mean is larger
            if np.random.rand() > 0.5:
                mean2 = mean1 + diff
            else:
                mean2 = mean1 - diff
            # Check both means are within [-1, 1]
            if -1 <= mean2 <= 1:
                std1 = np.random.uniform(0.05, 0.5)
                std2 = np.random.uniform(0.05, 0.5)
                vec1 = self.generate_vector(mean1, std1, length)
                vec2 = self.generate_vector(mean2, std2, length)
                return vec1, vec2

    def generate_dataset(self):
        ranges = [(0, 0.66), (0.66, 1.33), (1.33, 2.0)]
        vectors = []
        length = self.n_numbers

        for diff_range in ranges:
            for _ in range(60):
                v1, v2 = self.generate_vector_pair(diff_range, length)
                vectors.append((v1, v2))
                return v1, v2

    def record_solutions(self, v1, v2):
        """ Determine if answer is A, B, or C """
        mean1, mean2, diff = self.find_mean_difference(v1, v2)
        if mean1 > mean2:
            plot_val = 2 # for plotting
            answer = 'A' # X > Y
        elif mean1 < mean2:
            plot_val = 1 # for plotting
            answer = 'B' #  X < Y
        else:
            plot_val = 0 # for plotting
            answer = 'C' # Uncertain
        #if abs(mean1 - mean2) < epsilon:
        #    return 'C'  # Uncertain
        #elif mean1 > mean2:
        #    return 'A'
        #else:
        #    return 'B'
        #print(answer)
        return plot_val, answer


    def find_mean_difference(self, v1, v2):
        mean1 = np.mean(v1)
        mean2 = np.mean(v2)
        diff = abs(mean1 - mean2)
        return mean1, mean2, diff

    def assign_difficulty(self, v1, v2):
        mean1, mean2, diff_value = self.find_mean_difference(v1, v2)
        if diff_value <= self.difficulty_thresholds[0]:
            difficulty = 'hard'
        elif diff_value <= self.difficulty_thresholds[1]:
            difficulty = 'medium'
        elif diff_value > self.difficulty_thresholds[1]:
            difficulty = 'easy'
        else: # diff_value = NaN
            difficulty = 'N/A'    
        return difficulty


    def compute_confidence_intervals(vector_sets, confidence=0.95):
        z_score = 1.96  # for 95% confidence
        ci_results = []

        for idx, (v1, v2) in enumerate(vector_sets):
            mean1, std1 = np.mean(v1), np.std(v1, ddof=1)
            mean2, std2 = np.mean(v2), np.std(v2, ddof=1)
            n = len(v1)  # both vectors are of the same length

            ci1 = (mean1 - z_score * (std1 / np.sqrt(n)),
                   mean1 + z_score * (std1 / np.sqrt(n)))
            ci2 = (mean2 - z_score * (std2 / np.sqrt(n)),
                   mean2 + z_score * (std2 / np.sqrt(n)))

            ci_results.append({
                'pair_index': idx,
                'mean1': mean1,
                'ci1': ci1,
                'mean2': mean2,
                'ci2': ci2,
                'mean_diff': abs(mean1 - mean2)
            })
    
        return ci_results

    def compute_t_confidence_intervals(vector_sets, confidence=0.95):
        ci_results = []

        for idx, (v1, v2) in enumerate(vector_sets):
            n = len(v1)
            df = n - 1
            t_crit = t.ppf((1 + confidence) / 2, df)

            mean1, std1 = np.mean(v1), np.std(v1, ddof=1)
            mean2, std2 = np.mean(v2), np.std(v2, ddof=1)

            ci1 = (mean1 - t_crit * (std1 / np.sqrt(n)),
                mean1 + t_crit * (std1 / np.sqrt(n)))
            ci2 = (mean2 - t_crit * (std2 / np.sqrt(n)),
                mean2 + t_crit * (std2 / np.sqrt(n)))

            ci_results.append({
                'pair_index': idx,
                'mean1': mean1,
                'ci1': ci1,
                'mean2': mean2,
                'ci2': ci2,
                'mean_diff': abs(mean1 - mean2)
            })

        return ci_results

    def compute_bootstrap_confidence_intervals(vector_sets, confidence=0.95, n_bootstrap=1000, random_state=None):
        if random_state:
            np.random.seed(random_state)

        ci_results = []
        alpha = 1 - confidence
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        for idx, (v1, v2) in enumerate(vector_sets):
            means1 = [np.mean(np.random.choice(v1, size=len(v1), replace=True)) for _ in range(n_bootstrap)]
            means2 = [np.mean(np.random.choice(v2, size=len(v2), replace=True)) for _ in range(n_bootstrap)]

            ci1 = (np.percentile(means1, lower_percentile), np.percentile(means1, upper_percentile))
            ci2 = (np.percentile(means2, lower_percentile), np.percentile(means2, upper_percentile))

            mean1 = np.mean(v1)
            mean2 = np.mean(v2)

            ci_results.append({
                'pair_index': idx,
                'mean1': mean1,
                'ci1': ci1,
                'mean2': mean2,
                'ci2': ci2,
                'mean_diff': abs(mean1 - mean2)
            })

        return ci_results

    def classify_by_confidence_intervals(ci_results):
        A, B, C = [], [], []

        for result in ci_results:
            lower1, upper1 = result['ci1']
            lower2, upper2 = result['ci2']

            if lower1 > upper2:
                A.append(result)
            elif upper1 < lower2:
                B.append(result)
            else:
                C.append(result)

        return A, B, C


    def get_balanced_confidence_classification(ci_results, n_each=60, random_state=None):
        A, B, C = classify_by_confidence_intervals(ci_results)

        if random_state:
            random.seed(random_state)

        # Ensure we have enough in each category
        min_counts = min(len(A), len(B), len(C))
        if min_counts < n_each:
            raise ValueError(f"Not enough samples in each category to get {n_each} per class.")

        selected = random.sample(A, n_each) + random.sample(B, n_each) + random.sample(C, n_each)
        random.shuffle(selected)  # Shuffle for randomness

        return selected

    # Step 1: Generate bootstrap confidence intervals
    #bootstrap_ci_data = compute_bootstrap_confidence_intervals(vector_sets, n_bootstrap=1000, random_state=42)

    # Step 2: Get a balanced distribution
    #balanced_results = get_balanced_confidence_classification(bootstrap_ci_data, n_each=60, random_state=123)

    # Step 3: Count how many of each class we got
    #greater, less, uncertain = classify_by_confidence_intervals(balanced_results)
    #print(f"Greater: {len(greater)}, Less: {len(less)}, Uncertain: {len(uncertain)}")


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
