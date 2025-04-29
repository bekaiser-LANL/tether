"""This module defines simpleInequality benchmark that
generates two vector from a gaussian distribution
with and asks LLM which vector has the largest mean with X% confidence"""
import numpy as np
import matplotlib.pyplot as plt
from source.utils import QuestionBank
from source.utils import is_divisible_by_9


class SimpleInequality():
    """Generates questions about the simple inequality case to be saved 
    and then fed to LLMs"""

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
        chosen_range, vector_1, vector_2, _, _ = self.generate_dataset()

        # Convert the list of numbers to a space-separated string
        v1numbers_str = " ".join(map(str, vector_1))
        v2numbers_str = " ".join(map(str, vector_2))
        question = []
        question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
        Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
        Answer 'A' for yes, 'B' for no, or 'C' for uncertain. 
        Use only the data provided here and the 95% confidence level. 
        Do not repeat the prompt. Answer:"""
        return vector_1, vector_2, question, chosen_range

    def make_plot(self,count,vector_1,vector_2):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            import seaborn as sns
            sns.histplot(vector_1, kde=True, label="Vector 1", color="blue")
            sns.histplot(vector_2, kde=True, label="Vector 2", color="orange")
            plt.axvline(np.mean(vector_1), color='blue', linestyle='--')
            plt.axvline(np.mean(vector_2), color='orange', linestyle='--')
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
            # these range over varied n_samples:
            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)
            difficulty_tmp = np.empty(self.n_samples, dtype=object)
            n_samples_tmp = np.zeros([self.n_samples])
            mean_diff_tmp = np.zeros([self.n_samples])
            ci_lower_tmp = np.zeros([self.n_samples])
            ci_upper_tmp = np.zeros([self.n_samples])
            for i in reversed(range(self.n_samples)):

                #get questions:
                questions_tmp[i] = self.get_prompts()[2]
                #calculate the difficulty level
                difficulty_tmp[i] = self.assign_difficulty(
                        self.get_prompts()[0],
                        self.get_prompts()[1]
                        )
                #get CI bounds
                mean_diff_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[3]
                        )[2]
                ci_lower_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[3]
                        )[0]
                ci_upper_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[3]
                        )[1]
                #record the solutions:
                answers_tmp[i] = self.record_solutions(
                        ci_lower_tmp[i],
                        ci_upper_tmp[i]
                        )[1]
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
                "name": self.exam_name
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
                return vec1, vec2, std1, std2

    def generate_dataset(self):
        """Generate vector pairs with mean differences in ranges specified"""
        chosen_range = self.mean_diff_ranges[self.range_index]
        # Cycle through 0, 1, 2, 0, 1, ...
        self.range_index = (self.range_index + 1) % len(self.mean_diff_ranges)
        vector_1, vector_2, std1, std2 = self.generate_vector_pair(chosen_range)
        return chosen_range, vector_1, vector_2, std1, std2

    def calculate_ci(self, vector_1, vector_2, mean_diff_range):
        """Calculate the 95% confidence intervals around the means"""
        _, _, diff = self.find_mean_difference(vector_1, vector_2)
        _, _, std1, std2 = self.generate_vector_pair(mean_diff_range)
        std_error = np.sqrt(std1**2 / self.n_numbers + std2**2 / self.n_numbers)
        ci_upper = diff + 1.96*std_error
        ci_lower = diff - 1.96*std_error
        return ci_lower, ci_upper, diff

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
        diff = abs(mean1 - mean2)
        return mean1, mean2, diff

    def assign_difficulty(self, vector_1, vector_2):
        """Assign difficulty of problem based on mean differences"""
        _, _, diff_value = self.find_mean_difference(vector_1, vector_2)
        if diff_value <= self.difficulty_thresholds[0]:
            difficulty = 'hard'
        elif diff_value <= self.difficulty_thresholds[1]:
            difficulty = 'medium'
        elif diff_value > self.difficulty_thresholds[1]:
            difficulty = 'easy'
        else: # diff_value = NaN
            difficulty = 'N/A'
        return difficulty
