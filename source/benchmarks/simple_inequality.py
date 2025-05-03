"""This module defines simpleInequality benchmark that
generates two vector from a gaussian distribution
with and asks LLM which vector has the largest mean with X% confidence"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from source.utils import QuestionBank
from source.utils import is_divisible_by_9

class SimpleInequality():
    """Generates questions about the simple inequality case to be saved
    and then fed to LLMs"""

    def __init__(self, plot_path, exam_name, n_numbers = 100, **kwargs):

        self.plot_path = plot_path
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
        self.n_per_range = kwargs.get(
            'n_per_range',
            self.n_problems/len(self.answer_proportions)
        )
        self.n_per_range = int(self.n_per_range)
        self.n_samples = kwargs.get(
            'n_samples',
            self.n_problems/len(self.answer_proportions)
        )
        self.n_samples = int(self.n_samples)
        self.difficulty_thresholds = kwargs.get(
            'difficulty_thresholds',
            np.array([0.66,1.33])
        )
        try:
            self.ci_method = (exam_name).split('_')[1]
        except IndexError:
            self.ci_method = None
        self.exam_name_wo_ci_method = (exam_name).split('_')[0]
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
        chosen_range, vector_1, vector_2, std1, std2 = self.generate_dataset()

        # Convert the list of numbers to a space-separated string
        v1numbers_str = " ".join(map(str, vector_1))
        v2numbers_str = " ".join(map(str, vector_2))
        question = []
        if self.exam_name_wo_ci_method == 'SimpleInequality':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Answer 'A' for yes, 'B' for no, or 'C' for uncertain. 
            Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        if self.exam_name == 'SimpleInequalityWithMethod_tdist':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Calculate the 95% confidence level intervals. Use the 95% confidence intervals
            to answer 'A' for yes, 'B' for no, or 'C' for uncertain.
            Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        if self.exam_name == 'SimpleInequalityWithMethod_bootstrap':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Use bootstrap resampling to calculate the 95% confidence levels.
            Use the 95% confidence intervals to answer 'A' for yes, 'B' for no,
            or 'C' for uncertain. Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        return vector_1, vector_2, question, chosen_range, std1, std2

    def make_plot(self,problem, vector_1,vector_2):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            fig = plt.figure(figsize=(6, 5))
            mean_1 = np.mean(vector_1)
            std_1 = np.std(vector_1)
            xaxis_1 = np.linspace(mean_1 - 4*std_1, mean_1 + 4*std_1, 1000)
            gauss_1 = norm.pdf(xaxis_1, loc=mean_1, scale=std_1)
            mean_2 = np.mean(vector_2)
            std_2 = np.std(vector_2)
            xaxis_2 = np.linspace(mean_2 - 4*std_2, mean_2 + 4*std_2, 1000)
            gauss_2 = norm.pdf(xaxis_2, loc=mean_2, scale=std_2)
            ax_1 = sns.histplot(vector_1, color="blue", label='Sample 1')
            ax_2 = sns.histplot(vector_2, color="orange", label = 'Sample 2')
            ymax_1 = max([bar.get_height() for bar in ax_1.patches])
            ymax_2 = max([bar.get_height() for bar in ax_2.patches])
            gauss_1_scaled = gauss_1 * ymax_1 / max(gauss_1)  # scale to match histogram height
            gauss_2_scaled = gauss_2 * ymax_2 / max(gauss_2)
            plt.plot(xaxis_1, gauss_1_scaled, label ='Population Distribution 1')
            plt.plot(xaxis_2, gauss_2_scaled, label ='Population Distribution 2')
            plt.axvline(np.mean(vector_1), color='#56B4E9', linestyle='--')
            plt.axvline(np.mean(vector_2), color='#D55E00', linestyle='--')

            plt.legend()
            plot_name = f"{self.plot_path}/example_{problem['example_idx']}.png"
            plt.savefig(plot_name)
            plt.close(fig)

    def make_problems(self):
        """ Generate simple Inequality questions for the LLMs """

        qb = QuestionBank(target_per_bin =int(self.n_problems/9))
        test_complete = False
        example_idx = 0
        while not test_complete:
            # these range over varied n_samples:
            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)
            difficulty_tmp = np.empty(self.n_samples, dtype=object)
            n_samples_tmp = np.zeros([self.n_samples])
            mean_diff_tmp = np.zeros([self.n_samples])
            ci_lower_tmp = np.zeros([self.n_samples])
            ci_upper_tmp = np.zeros([self.n_samples])
            vectors_1 = np.empty((self.n_samples, self.n_numbers))
            vectors_2 = np.empty((self.n_samples, self.n_numbers))
            for i in reversed(range(self.n_samples)):
                vec1, vec2, question, chosen_range, std1, std2 = self.get_prompts()

                vectors_1[i, :] = vec1
                vectors_2[i, :] = vec2
                questions_tmp[i] = question

                if self.ci_method == 'tdist':
                    ci_lower, ci_upper, diff = self.calculate_ci(vec1, vec2, std1, std2)
                elif self.ci_method == 'bootstrap':
                    ci_lower, ci_upper, diff = self.bootstrap_ci(vec1, vec2)

                mean_diff_tmp[i] = diff
                ci_lower_tmp[i] = ci_lower
                ci_upper_tmp[i] = ci_upper
                answers_tmp[i] = self.record_solutions(ci_lower, ci_upper)[1]
                difficulty_tmp[i] = self.assign_difficulty(diff)

            # Randomly select one case from the generated examples
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
                "mean_diff": mean_diff_tmp[subsample_idx],
                "ci_lower": ci_lower_tmp[subsample_idx],
                "ci_upper": ci_upper_tmp[subsample_idx],
                "n_samples": n_samples_tmp[subsample_idx],
                "mean_diff_all": mean_diff_tmp,
                "ci_lower_all": ci_lower_tmp,
                "ci_upper_all": ci_upper_tmp,
                "n_samples_all": n_samples_tmp,
                "subsample_idx": subsample_idx,
                "example_idx": example_idx,
                "name": self.exam_name,
                "vector_1": vectors_1[subsample_idx],
                "vector_2": vectors_2[subsample_idx],
            }

            if self.verbose:
                print('\n mean_diff = ',problem["mean_diff"])

            if qb.add_question(
                problem["question"],
                problem["solution"],
                problem["difficulty"],
                metadata={
                    k: v
                    for k, v in problem.items()
                    if k not in {"question", "solution", "difficulty"}}
                ):
                self.make_plot(problem,
                               problem["vector_1"],
                               problem["vector_2"])
            print(qb.count())
            # Check if ready:
            if qb.is_full():
                final_set = qb.get_balanced_set()
                if self.verbose:
                    print("Test is complete:", len(final_set), "questions")
                test_complete = True
                #Pull attributes from qb
                qb.mean_diff = np.array([
                    q['metadata']['mean_diff'] for q in qb.get_balanced_set()]
                )
                qb.ci_lower = np.array([
                    q['metadata']['ci_lower'] for q in qb.get_balanced_set()]
                )
                qb.ci_upper = np.array([
                    q['metadata']['ci_upper'] for q in qb.get_balanced_set()]
                )
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
        vec = np.random.normal(loc=target_mean, scale=target_std, size=length)
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
            ## Check both means are within [-1, 1]
            if -1 <= mean2 <= 1:
                std1 = np.random.uniform(0.5, 8)
                std2 = np.random.uniform(0.5, 8)
                vec1 = self.generate_vector(mean1, std1, length)
                vec2 = self.generate_vector(mean2, std2, length)
                return vec1, vec2, std1, std2


    def generate_dataset(self):
        """Generate vector pairs with mean differences in ranges specified"""
        chosen_range = self.mean_diff_ranges[self.range_index]
        # Cycle through 0, 1, 2, 0, 1, ...
        self.range_index = (self.range_index + 1) % len(self.mean_diff_ranges)
        vector_1, vector_2, std1, std2 = self.generate_vector_pair(chosen_range)
        return chosen_range, vector_1, vector_2, std1, std2

    def calculate_ci(self, vector_1, vector_2, std1, std2):
        """Calculate the 95% confidence intervals around the means"""
        _, _, diff = self.find_mean_difference(vector_1, vector_2)
        std_error = np.sqrt(std1**2 / self.n_numbers + std2**2 / self.n_numbers)
        ci_upper = diff + 1.96*std_error
        ci_lower = diff - 1.96*std_error
        return ci_lower, ci_upper, diff

    def bootstrap_ci(self, vector_1, vector_2, n_bootstrap=2000, ci=95):
        """
        Bootstrap the mean differences to estimate confidence intervals.

            Args:
                vector_1 (np.ndarray): First sample.
                vector_2 (np.ndarray): Second sample.
                n_bootstrap (int): Number of bootstrap resamples.
                ci (float): Confidence level (default 95).

            Returns:
                (ci_lower, ci_upper, observed_diff)
        """
        rng = np.random.default_rng()

        # Preallocate resampling indices
        idx_1 = rng.integers(0, len(vector_1), size=(n_bootstrap, len(vector_1)))
        idx_2 = rng.integers(0, len(vector_2), size=(n_bootstrap, len(vector_2)))

        # Use indexing to generate resampled arrays
        resampled_1 = vector_1[idx_1]
        resampled_2 = vector_2[idx_2]

        # Compute mean differences all at once
        diffs = resampled_1.mean(axis=1) - resampled_2.mean(axis=1)

        # Observed diff
        observed_diff = np.mean(vector_1) - np.mean(vector_2)

        # Confidence interval
        lower, upper = np.percentile(diffs, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        return lower, upper, observed_diff


    def record_solutions(self, ci_lower, ci_upper):
        """ Determine if answer is A, B, or C """
        if ci_lower > 0:
            plot_val = 2 # for plotting
            answer = 'A' # X > Y
        elif ci_upper < 0:
            plot_val = 1 # for plotting
            answer = 'B' #  X < Y
        else:
            plot_val = 0 # for plotting
            answer = 'C' # Uncertain
        return plot_val, answer

    def find_mean_difference(self, vector_1, vector_2):
        """Calculate the difference between each vector mean"""
        mean1 = np.mean(vector_1)
        mean2 = np.mean(vector_2)
        diff = mean1 - mean2
        return mean1, mean2, diff

    def assign_difficulty(self, diff_value):
        """Assign difficulty of problem based on mean differences"""
        if abs(diff_value) <= self.difficulty_thresholds[0]:
            difficulty = 'hard'
        elif abs(diff_value) <= self.difficulty_thresholds[1]:
            difficulty = 'medium'
        elif abs(diff_value) > self.difficulty_thresholds[1]:
            difficulty = 'easy'
        else: # diff_value = NaN
            difficulty = 'N/A'
        return difficulty
