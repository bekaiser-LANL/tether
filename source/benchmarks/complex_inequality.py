"""This module defines ComplexInequality benchmark that
generates two vector from a multimodel distribution
with and asks LLM which vector has the largest mean with X% confidence"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.integrate import trapezoid
from scipy.stats import norm
from scipy.interpolate import interp1d
from source.utils import QuestionBank
from source.utils import is_divisible_by_9

class ComplexInequality():
    """Generates questions about the complex inequality case to be saved and then fed to LLMs"""

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
        if self.exam_name_wo_ci_method == 'ComplexInequality':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Answer 'A' for yes, 'B' for no, or 'C' for uncertain. 
            Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        if self.exam_name == 'ComplexInequalityWithMethod_tdist':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Calculate the 95% confidence level intervals. Use the 95% confidence intervals
            to answer 'A' for yes, 'B' for no, or 'C' for uncertain.
            Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        if self.exam_name == 'ComplexInequalityWithMethod_bootstrap':
            question = f"""Vector 1: {v1numbers_str} Vector 2: {v2numbers_str}
            Is it more probable that a sample from Vector 1 is greater than sample from Vector 2? 
            Use bootstrap resampling to calculate the 95% confidence levels.
            Use the 95% confidence intervals to answer 'A' for yes, 'B' for no,
            or 'C' for uncertain. Use only the data provided here and the 95% confidence level. 
            Do not repeat the prompt. Answer:"""
        return vector_1, vector_2, question, chosen_range, std1, std2

    def make_plot(self):
        if self.plot_flag: # make a plot of the 95% confidence interval
            n_points = 10000
            x_low = -15
            x_high = 15
            x_vec = np.linspace(x_low, x_high, n_points)

            #figname = './many_dists.png'
            #fig = plt.figure(figsize=(6, 5))
            #ax1=plt.subplot(1,1,1)
            #y_vec = self.multimodal_pdf(x_vec, -5, 5, 0.5, 2., -20, 20, n_points)
            #print(trapezoid(y, x))
            #plt.plot(x_vec, y_vec)
            #plt.title("Skewed Gaussian Distribution")
            #plt.xlabel("x")
            #plt.ylabel("Density")
            #plt.grid(True)
            #plt.xlim([-7.5, 7.5])
            #plt.subplots_adjust(
            #        top=0.95, bottom=0.14, left=0.15,
            #        right=0.985, hspace=0.4, wspace=0.35)
            #plt.savefig(figname,format="png")
            #plt.close(fig)

            y_vec = self.multimodal_pdf(x_vec, -5, 5, 0.5, 2., -20, 20, n_points)

            figname = './one_dist.png'
            fig = plt.figure(figsize=(6, 5))
            #ax1=plt.subplot(1,1,1)
            plt.plot(x_vec, y_vec)
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.grid(True)
            plt.xlim([-7.5, 7.5])
            plt.subplots_adjust(
                    top=0.95, bottom=0.14, left=0.15,
                    right=0.985, hspace=0.4, wspace=0.35)
            plt.savefig(figname,format="png")
            plt.close(fig)

            prob_0 = trapezoid(self.gaussian(x_vec),x_vec)
            prob_1 = trapezoid(y_vec, x_vec)
            prob_2 = self.prob_greater_than_from_pdf(x_vec, y_vec, -500.)
            prob_3 = self.prob_greater_than_from_pdf(x_vec, y_vec, 0.)
            prob_4 = self.prob_greater_than_from_pdf(x_vec, self.gaussian(x_vec), 0.)
            print('\n a) gaussian: Should be 1 (integrate population density): ',prob_0)
            print('\n b) one_dist: Should be 1 (integrate population density): ',prob_1)
            print('\n c) one_dist: Should be ~1 (integrate sample density):',prob_2)
            print('\n d) one_dist: Look at one_dist.png - Probability P(x>0): ',prob_3)
            print('\n e) gaussian: Should be 0.5 - Probability P(x>0): ',prob_4)

            pdf_func = interp1d(x_vec, y_vec, bounds_error=False, fill_value=0.0)
            samples = self.rejection_sample(
                    pdf_func, x_min=x_low, x_max=x_high, y_max=np.amax(y_vec), n_samples=10000)
            # samples are effectively normalized frequencies, meaning you could have
            # n number of samples in 0.1<x<0.2, and what is output here would be n/n_samples
            # where n_samples is the total number of samples.
            # Note that y_max=np.amax(y) when we know y a priori.

            figname = './one_dist_samples.png'
            fig = plt.figure(figsize=(6, 5))
            #ax1=plt.subplot(1,1,1)
            plt.plot(x_vec, y_vec)
            plt.hist(samples, bins=50, density=True, alpha=0.6, label="Sample Histogram")
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.grid(True)
            plt.xlim([-7.5, 7.5])
            plt.subplots_adjust(
                    top=0.95, bottom=0.14, left=0.15, right=0.985, hspace=0.4, wspace=0.35)
            plt.savefig(figname,format="png")
            plt.close(fig)

            prob_5 = self.prob_greater_than_x0_from_samples(samples, 0.)
            print('\n f) one_dist: Sample integration\n,'
                    'should be close to d) - Probability P(x>0): ',prob_5)

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
                if self.ci_method == 'tdist':
                    mean_diff_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[4],
                        self.get_prompts()[5]
                        )[2]
                    ci_lower_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[4],
                        self.get_prompts()[5]
                        )[0]
                    ci_upper_tmp[i] = self.calculate_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1],
                        self.get_prompts()[4],
                        self.get_prompts()[5]
                        )[1]
                if self.ci_method == 'bootstrap':
                    mean_diff_tmp[i] = self.bootstrap_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1]
                        )[2]
                    ci_lower_tmp[i] = self.bootstrap_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1]
                        )[0]
                    ci_upper_tmp[i] = self.bootstrap_ci(
                        self.get_prompts()[0],
                        self.get_prompts()[1]
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
                self.make_plot()
                count = count + 1
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

    def generate_vector(self, x_low, x_high, mean_lower,
                        mean_upper, std_lower, std_upper, alpha_lower, alpha_upper, n_points):
        """Create vector of length n_numbers sampled from multimodal pdf"""
        x_vec = np.linspace(x_low, x_high, n_points)
        y_vec = self.multimodal_pdf(x_vec, mean_lower, mean_upper,
                                    std_lower, std_upper, alpha_lower, alpha_upper, n_points)
        length = self.n_numbers
        pdf_normalized = y_vec/np.sum(y_vec)
        vec = np.random.choice(x_vec,size=length,p=pdf_normalized)
        return vec

    def generate_vector_pair(self, mean_diff_range):
        """Generate vectors with random means and stdevs in given ranges"""
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
                #(x_low, x_high, mean_lower, mean_upper, std_lower,
                #std_upper, alpha_lower, alpha_upper, N)
                vec1 = self.generate_vector(-15, 15, -5, 5, 0.5, 2., -20, 20, 1000)
                vec2 = self.generate_vector(-15, 15, -5, 5, 0.5, 2., -20, 20, 1000)
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

    def bootstrap_ci(self, vector_1, vector_2, conf_int=95):
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
        rng = np.random.default_rng()  # Faster random generator (numpy 1.17+)
        n_bootstrap = self.n_bootstrap
        diffs = np.zeros(n_bootstrap)

        for i in range(n_bootstrap):
            sample_1 = rng.choice(vector_1, size=len(vector_1), replace=True)
            sample_2 = rng.choice(vector_2, size=len(vector_2), replace=True)
            diffs[i] = np.mean(sample_1) - np.mean(sample_2)

        # Compute observed difference
        observed_diff = np.mean(vector_1) - np.mean(vector_2)

        # Get percentiles
        lower_percentile = (100 - conf_int) / 2
        upper_percentile = 100 - lower_percentile

        ci_lower = np.percentile(diffs, lower_percentile)
        ci_upper = np.percentile(diffs, upper_percentile)
        return ci_lower, ci_upper, observed_diff

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

    def superposed_sine_waves(self, n_waves=20, duration=1.0, sample_rate=1000,
                          freq_range=(1, 40), amp_range=(0.5, 1.0), seed=None):
        """
        Generate a superposition of sine waves with random amplitudes and frequencies.

        Parameters:
            n_waves (int): Number of sine waves to superimpose.
            duration (float): Duration of the signal in seconds.
            sample_rate (int): Number of samples per second.
            freq_range (tuple): Frequency range (min, max) in Hz.
            amp_range (tuple): Amplitude range (min, max).
            seed (int or None): Random seed for reproducibility.
        Returns:
            t (np.ndarray): Time array.
            signal (np.ndarray): Superposed sine wave signal.
        """
        if seed is not None:
            np.random.seed(seed)

        t_vec = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        signal = np.zeros_like(t_vec)

        for _ in range(n_waves):
            freq = np.random.uniform(*freq_range)
            amp = np.random.uniform(*amp_range)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * t_vec + phase)

        return np.abs(signal)

    def multimodal_pdf(self, x_vec, mean_lower, mean_upper,
                       std_lower, std_upper, alpha_lower, alpha_upper, n_points):
        """Normalized multi-modal PDF with compact support"""
        mean = np.random.uniform(mean_lower, mean_upper)
        std = np.random.uniform(std_lower, std_upper)
        alpha = np.random.uniform(alpha_lower, alpha_upper)
        raw = self.superposed_sine_waves(sample_rate=n_points) * self.skewed_gaussian(
                x_vec, mean=mean, std=std, alpha=alpha)
        #raw = skewed_gaussian(x, mean=mean, std=std, alpha=alpha)
        area = trapezoid(raw, x_vec)
        return raw / area if area > 0 else np.zeros_like(x_vec)

    def skewed_gaussian(self, x_vec, mean=0, std=1, alpha=0):
        """
        Returns a skewed Gaussian (skew-normal) curve.

        Parameters:
        - x: input array
        - mean: center of the distribution
        - std: standard deviation (spread)
        - alpha: skewness parameter (0 = normal; >0 = right skew; <0 = left skew)

        Returns:
        - Skewed Gaussian values at x
        """
        t_vec = (x_vec - mean) / std
        pdf = norm.pdf(t_vec)
        cdf = norm.cdf(alpha * t_vec)
        return 2 * pdf * cdf

    def prob_greater_than_x0_from_samples(self, samples, x0_vec, bins=1000):
        # Get histogram (density=True normalizes it)
        counts, bin_edges = np.histogram(samples, bins=bins, density=True)
        bin_widths = np.diff(bin_edges)

        # Bin centers (optional)
        #bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate the area (probability) for each bin
        bin_probs = counts * bin_widths  # height * width = area

        # Get mask of bins where upper edge > x0
        mask = bin_edges[:-1] > x0_vec  # lower edge of each bin is bin_edges[i]

        # Handle edge case: partial bin containing x0
        first_bin_idx = np.searchsorted(bin_edges, x0_vec, side='right') - 1
        if 0 <= first_bin_idx < len(bin_probs):
            # Fractional contribution from the first bin
            #bin_start = bin_edges[first_bin_idx]
            bin_end = bin_edges[first_bin_idx + 1]
            #fraction = (bin_end - x0) / (bin_end - bin_start)
            partial_area = counts[first_bin_idx] * (bin_end - x0_vec)
        else:
            partial_area = 0.0

        # Sum up remaining full bins
        prob = np.sum(bin_probs[mask]) + partial_area

        return prob

    def gaussian(self, x_vec, mean=0, std=1):
        """construct a normal distribution"""
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x_vec - mean) ** 2) / (2 * std ** 2))

    def prob_greater_than_from_pdf(self, x_vec, pdf_values, x0_vec):
        mask = x_vec > x0_vec
        return np.trapz(pdf_values[mask], x_vec[mask])

    def rejection_sample(self, pdf_func, x_min, x_max, y_max, n_samples):
        """ We know our pdf as a function of x. We don't know if
        its cdf is invertible a priori. Therefore, let's use
        rejection sampling to sample it. """
        samples = []
        while len(samples) < n_samples:
            x_vec = np.random.uniform(x_min, x_max)
            y_vec = np.random.uniform(0, y_max)
            if y_vec <= pdf_func(x_vec):
                samples.append(x_vec)
        return np.array(samples)
