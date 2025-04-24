"""This module defines simpleInequality benchmark that 
generates two vector from a gaussian distribution
with and asks LLM which vector has the largest mean with X% confidence"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t
from source.utils import QuestionBank
from source.utils import is_divisible_by_9

class SimpleInequality():
    """Generates questions about the simple inequality case to be saved and then fed to LLMs"""

    def __init__(self, exam_name, n_numbers = 100, **kwargs):

        #self.plot_path = plot_path
        self.exam_name = exam_name
        #generation parameters:
        self.n_problems = kwargs.get('n_problems', 18) #length of test
        self.n_numbers = n_numbers #length of each vector
        self.plot_flag = kwargs.get('plot_flag', False)
        self.generate_flag = kwargs.get('generate_flag', True)
        self.verbose = kwargs.get('verbose', False)
        self.mean_diff_ranges = kwargs.get('mean_diff_ranges',[
            (0, 0.66),
            (0.66, 1.33),
            (1.33, 2.0)
        ])
        self.answer_proportions = kwargs.get(
            "answer_proportions",
            [0.333, 0.333, 0.333], # Ratios of A, B, C correct answers
        )
        self.n_per_range = kwargs.get('n_per_range', self.n_problems/len(self.answer_proportions))
        self.n_per_range = int(self.n_per_range)
        self.n_samples = kwargs.get('n_samples', self.n_problems/len(self.answer_proportions))
        self.n_samples = int(self.n_samples)
        self.difficulty_thresholds = kwargs.get(
            'difficulty_thresholds',
            np.array([0.66,1.33])
        )
        #self.ci_method = (exam_name).split('_')[1]
        #self.exam_name_wo_ci_method = (exam_name).split('_')[0]
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        self.range_index=0
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
        q = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
        Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
        Answer 'A' for yes, 'B' for no, or 'C' for uncertain. 
        Use only the data provided here and the 95% confidence level. 
        Do not repeat the prompt. Answer:"""
        return v1, v2, q

    def make_plot(self,count,v1,v2):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            import seaborn as sns
            sns.histplot(v1, kde=True, label="Vector 1", color="blue")
            sns.histplot(v2, kde=True, label="Vector 2", color="orange")
            plt.axvline(np.mean(v1), color='blue', linestyle='--')
            plt.axvline(np.mean(v2), color='orange', linestyle='--')
            plt.legend()
            plt.title("Distribution with KDE")
            plot_name = f"distribution_plot_{count}.png"
            plt.savefig(plot_name)

    def make_problems(self):
        """ Generate simple Inequality questions for the LLMs """

        qb = QuestionBank(target_per_bin =int(self.n_problems/9))
        test_complete = False
        example_idx = 0
        count = 0
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
                difficulty_tmp[i] = self.assign_difficulty(
                        self.get_prompts()[0],
                        self.get_prompts()[1]
                        )
                #record the solutions:
                answers_tmp[i] = self.record_solutions(
                        self.get_prompts()[0],
                        self.get_prompts()[1])[1]
                #print(answers_tmp[i])
                mean_diff_tmp[i] = self.find_mean_difference(
                        self.get_prompts()[0],
                        self.get_prompts()[1])[2]
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
                metadata={
                    k: v
                    for k, v in problem.items()
                    if k not in {"question", "solution", "difficulty"}}
                ):
                self.make_plot(count,self.get_prompts()[0],self.get_prompts()[1])
                count = count + 1

            # Check if ready:
            if qb.is_full():
                final_set = qb.get_balanced_set()
                if self.verbose:
                    print("Test is complete:", len(final_set), "questions")
                test_complete = True
                #Pull attributes from qb
                qb.n_samples = np.array(
                        [q['metadata']['n_samples'] for q in qb.get_balanced_set()]
                        )
                qb.name = np.array(
                        [q['metadata']['name'] for q in qb.get_balanced_set()]
                        )
                qb.example_idx = np.array(
                        [q['metadata']['example_idx'] for q in qb.get_balanced_set()]
                        )
                qb.solution = [q['solution'] for q in qb.get_balanced_set()]
                qb.question = [q['question'] for q in qb.get_balanced_set()]
                qb.difficulty = [q['difficulty'] for q in qb.get_balanced_set()]
                for name, value in qb.__dict__.items():
                    setattr(self, name, value)
            else:
                if self.verbose:
                    print("Still building test. Current count:", qb.count())
                example_idx += 1 # loop over examples
        print('Done!')

    def generate_vector(self, target_mean, target_std, length):
        """Generate vector with gaussian centered about mean with std"""
        length = self.n_numbers
        vec = np.random.randn(length)
        vec -= np.mean(vec)
        vec /= np.std(vec)
        vec *= target_std
        vec += target_mean
        vec = np.round(vec, 2) #round to 2 decimal places
        return vec

    def generate_vector_pair(self, mean_diff_range):
        """Generate vectors with random means and stdevs in given ranges"""
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
        """Generate vector pairs with mean differences in ranges specified"""
        chosen_range = self.mean_diff_ranges[self.range_index]
        # Cycle through 0, 1, 2, 0, 1, ...
        self.range_index = (self.range_index + 1) % len(self.mean_diff_ranges)
        return self.generate_vector_pair(chosen_range)

    def record_solutions(self, v1, v2):
        """ Determine if answer is A, B, or C """
        mean1, mean2, _ = self.find_mean_difference(v1, v2)
        if mean1 > mean2:
            plot_val = 2 # for plotting
            answer = 'A' # X > Y
        elif mean1 < mean2:
            plot_val = 1 # for plotting
            answer = 'B' #  X < Y
        else:
            plot_val = 0 # for plotting
            answer = 'C' # Uncertain
        return plot_val, answer


    def find_mean_difference(self, v1, v2):
        """Calculate the difference between each vector mean"""
        mean1 = np.mean(v1)
        mean2 = np.mean(v2)
        diff = abs(mean1 - mean2)
        return mean1, mean2, diff

    def assign_difficulty(self, v1, v2):
        """Assign difficulty of problem based on mean differences"""
        _, _, diff_value = self.find_mean_difference(v1, v2)
        if diff_value <= self.difficulty_thresholds[0]:
            difficulty = 'hard'
        elif diff_value <= self.difficulty_thresholds[1]:
            difficulty = 'medium'
        elif diff_value > self.difficulty_thresholds[1]:
            difficulty = 'easy'
        else: # diff_value = NaN
            difficulty = 'N/A'
        return difficulty

    def compute_confidence_intervals(self, vector_sets):
        """Population confidence interval (z-scores)"""
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

    def compute_t_confidence_intervals(self, vector_sets, confidence=0.95):
        """Calculate CI using student t"""
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

    def compute_bootstrap_confidence_intervals(
            self,
            vector_sets,
            confidence=0.95,
            n_bootstrap=1000,
            random_state=None
            ):
        """Use bootstrap method to find CIs"""
        if random_state:
            np.random.seed(random_state)

        ci_results = []
        alpha = 1 - confidence
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        for idx, (v1, v2) in enumerate(vector_sets):
            means1 = [
                    np.mean(np.random.choice(v1, size=len(v1), replace=True))
                    for _ in range(n_bootstrap)
                    ]
            means2 = [
                    np.mean(np.random.choice(v2, size=len(v2), replace=True))
                    for _ in range(n_bootstrap)
                    ]

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
