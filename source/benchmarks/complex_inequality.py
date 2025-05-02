"""This module defines ComplexInequality benchmark that
generates two vector from a multimodel distribution
with and asks LLM which vector has the largest mean with X% confidence"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.integrate import trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm, stats
from scipy.interpolate import interp1d
from source.utils import QuestionBank
from source.utils import is_divisible_by_9

class ComplexInequality():
    """Generates questions about the complex inequality case to be saved and then fed to LLMs"""

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
        chosen_range, vector_1, vector_2, std1, std2, xvec_1, yvec_1, xvec_2, yvec_2 = self.generate_dataset()

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
        return vector_1, vector_2, question, chosen_range, std1, std2, xvec_1, yvec_1, xvec_2, yvec_2

    def make_plot(self, problem, vector_1, vector_2, xvec_1, yvec_1, xvec_2, yvec_2):
        if self.plot_flag: # make a plot of the 95% confidence interval
            import seaborn as sns
            from scipy import stats
            from scipy.ndimage import gaussian_filter1d
            # Get sample modes
            #mode_1 = stats.mode(np.round(vector_1, 2), keepdims=True).mode[0]
            #mode_2 = stats.mode(np.round(vector_2, 2), keepdims=True).mode[0]
            
            # Optional smoothing (if not already applied)
            yvec_1 = gaussian_filter1d(yvec_1, sigma=2)
            yvec_2 = gaussian_filter1d(yvec_2, sigma=2)

            # Histogram counts and bin widths
            counts_1, bin_edges_1 = np.histogram(vector_1, bins=30)
            counts_2, bin_edges_2 = np.histogram(vector_2, bins=30)
            bin_width_1 = bin_edges_1[1] - bin_edges_1[0]
            bin_width_2 = bin_edges_2[1] - bin_edges_2[0]

            # Area-based scaling
            yvec_1_area_scaled = yvec_1 * len(vector_1) * bin_width_1
            yvec_2_area_scaled = yvec_2 * len(vector_2) * bin_width_2

            # Peak scaling to match histogram max
            max_hist_1 = np.max(counts_1)
            max_hist_2 = np.max(counts_2)

            peak_scale_1 = max_hist_1 / np.max(yvec_1_area_scaled)
            peak_scale_2 = max_hist_2 / np.max(yvec_2_area_scaled)

            yvec_1_scaled = yvec_1_area_scaled * peak_scale_1
            yvec_2_scaled = yvec_2_area_scaled * peak_scale_2

            # Compute histogram mode bin center
            max_bin_idx_1 = np.argmax(counts_1)
            max_bin_idx_2 = np.argmax(counts_2)
            mode_1 = (bin_edges_1[max_bin_idx_1] + bin_edges_1[max_bin_idx_1 + 1]) / 2
            mode_2 = (bin_edges_2[max_bin_idx_2] + bin_edges_2[max_bin_idx_2 + 1]) / 2

            # PDF mode alignment
            pdf_mode_1 = xvec_1[np.argmax(yvec_1)]
            pdf_mode_2 = xvec_2[np.argmax(yvec_2)]
            shift_1 = mode_1 - pdf_mode_1
            shift_2 = mode_2 - pdf_mode_2

            xvec_1_shifted = xvec_1 + shift_1
            xvec_2_shifted = xvec_2 + shift_2 
            # Plot
            fig, ax = plt.subplots(figsize=(7, 5))

            sns.histplot(vector_1, bins=30, color='blue', label='Sample 1')
            sns.histplot(vector_2, bins=30, color='orange', label='Sample 2')

            plt.plot(xvec_1_shifted, yvec_1_scaled, color='blue', label='Population Distribution 1')
            plt.plot(xvec_2_shifted, yvec_2_scaled, color='orange', label='Population Distribution 2')

            plt.axvline(mode_1, color='#56B4E9', linestyle='--')
            plt.axvline(mode_2, color='#D55E00', linestyle='--')
            ymax = max(np.max(counts_1), np.max(counts_2), np.max(yvec_1_scaled), np.max(yvec_2_scaled))

            plt.ylim(top=ymax * 1.1)  # Add 10% padding above
            figname = f"{self.plot_path}/example_{problem['example_idx']}.png"
            plt.legend()
            plt.savefig(figname)
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
            xvecs_1 = np.empty((self.n_samples, 1000))
            yvecs_1 = np.empty((self.n_samples, 1000))
            xvecs_2 = np.empty((self.n_samples, 1000))
            yvecs_2 = np.empty((self.n_samples, 1000))
 
            for i in reversed(range(self.n_samples)):
                vec1, vec2, question, chosen_range, std1, std2, xvec1, yvec1, xvec2, yvec2 = self.get_prompts()

                vectors_1[i, :] = vec1
                vectors_2[i, :] = vec2
                xvecs_1[i, :] = xvec1
                yvecs_1[i, :] = yvec1
                xvecs_2[i, :] = xvec2
                yvecs_2[i, :] = yvec2
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
                "xvec_1": xvecs_1[subsample_idx],
                "yvec_1": yvecs_1[subsample_idx],
                "xvec_2": xvecs_2[subsample_idx],
                "yvec_2": yvecs_2[subsample_idx]
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
                                problem["vector_2"],
                                problem["xvec_1"],
                                problem["yvec_1"],
                                problem["xvec_2"],
                                problem["yvec_2"])
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

    def generate_vector(self, x_low, x_high, n_points,target_mode=None, target_std=None):
        """Create vector of length n_numbers sampled from multimodal pdf with target mode&std"""
        x_vec = np.linspace(x_low, x_high, n_points)
        y_vec = self.multimodal_pdf(x_vec, n_points)
        length = self.n_numbers
        if target_mode is not None:
            x_vec, y_vec = self.shift_pdf(x_vec, y_vec, target_mode)
        pdf_normalized = y_vec/np.sum(y_vec)
        vec = np.random.choice(x_vec,size=length,p=pdf_normalized)
        if target_std is not None:
            # Normalize to target mean and std
            vec = (vec - np.mean(vec)) / np.std(vec)
            vec = vec * target_std + np.mean(vec)
        return vec, x_vec, y_vec
    
    def shift_pdf(self, x_vec, y_vec, target_mode):
        """Shift PDF to center its mode at the target_mode."""
        current_mode = x_vec[np.argmax(y_vec)]
        shift = target_mode - current_mode
        return x_vec + shift, y_vec

    def generate_vector_pair(self, mode_diff_range):
        """Generate vector pair with target mode difference and controlled std."""
        attempts = 0
        max_attempts = 100

        while attempts < max_attempts:
            # Pick a target mean/mode difference
            mode1 = np.random.uniform(-1,1)
            diff = np.random.uniform(*mode_diff_range)

            if np.random.rand() > 0.5:
                mode2 = mode1 + diff
            else:
                mode2 = mode1 - diff

            if -1 <= mode2 <=1:
                # Choose stds in a reasonable range
                std1 = np.random.uniform(0.3, 8)
                std2 = np.random.uniform(0.3, 8)

                # Generate vectors centered and scaled exactly
                vec1, xvec_1, yvec_1 = self.generate_vector(-10, 10, 1000, target_mode=mode1, target_std=std1)
                vec2, xvec_2, yvec_2 = self.generate_vector(-10, 10, 1000, target_mode=mode2, target_std=std2)
                return vec1, vec2, std1, std2, xvec_1, yvec_1, xvec_2, yvec_2


    def generate_dataset(self):
        """Generate vector pairs with mode differences in ranges specified"""
        chosen_range = self.mean_diff_ranges[self.range_index]
        # Cycle through 0, 1, 2, 0, 1, ...
        self.range_index = (self.range_index + 1) % len(self.mean_diff_ranges)
        vector_1, vector_2, std1, std2, xvec_1, yvec_1, xvec_2, yvec_2 = self.generate_vector_pair(chosen_range)
        return chosen_range, vector_1, vector_2, std1, std2, xvec_1, yvec_1, xvec_2, yvec_2

    def superposed_sine_waves(self, x_vec, n_waves=3, duration=1.0, sample_rate=1000,
                          freq_range=(0.5, 2.0), amp_range=(0.5, 1.0), seed=None):
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

        signal = np.zeros_like(x_vec)

        for _ in range(n_waves):
            freq = np.random.uniform(*freq_range)
            amp = np.random.uniform(*amp_range)
            phase = np.random.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * freq * x_vec + phase)

        return np.abs(signal)

    def multimodal_pdf(self, x_vec, n_points):
        """Create a clean, smooth multimodal PDF using a mixture of Gaussians."""
        y = np.zeros_like(x_vec)

        n_components = np.random.randint(6,11) #np.random.choice([2, 3, 4])
        centers = np.linspace(-6, 6, n_components) + np.random.normal(0, 0.3, n_components)

        for c in centers:
            std = np.random.uniform(0.3, 1.0)
            height = np.random.uniform(0.6, 1.2)
            y += height * norm.pdf(x_vec, loc=c, scale=std)

        # Add slight undulation: low-freq sine wave
        # Add slight undulation with low-frequency wiggle
        #phase = np.random.uniform(0, 2 * np.pi)
        #wiggle = 0.2 * np.sin(2 * np.pi * x_vec / 5 + phase)
        #y *= (1 + wiggle)

        # Optional smoothing to clean sharp transitions
        y = gaussian_filter1d(y, sigma=0.5)
        
        # Normalize so the PDF integrates to 1
        area = trapezoid(y, x_vec)
        return y / area if area > 0 else np.zeros_like(x_vec)


    def old_multimodal_pdf(self, x_vec, n_points):
        """Normalized multi-modal PDF with compact support"""
        mean = np.random.uniform(-1, 1)
        std = np.random.uniform(0.05, 1.5)
        alpha = np.random.uniform(-20, 20)
        raw = self.superposed_sine_waves(x_vec, sample_rate=n_points) * self.skewed_gaussian(
                x_vec, mean=mean, std=std, alpha=alpha)
        raw_smoothed = gaussian_filter1d(raw, sigma=4)
        #raw = skewed_gaussian(x, mean=mean, std=std, alpha=alpha)
        area = trapezoid(raw_smoothed, x_vec)
        return raw_smoothed / area if area > 0 else np.zeros_like(x_vec)

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
        return trapezoid(pdf_values[mask], x_vec[mask])

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
    
    def calculate_ci(self, vector_1, vector_2, std1, std2):
        """Calculate the 95% confidence intervals around the means"""
        _, _, diff = self.find_mode_difference(vector_1, vector_2)
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

    def find_mode_difference(self, vector_1, vector_2):
        """Calculate the difference between each vector mean"""
        mode1 = stats.mode(vector_1, keepdims=True).mode[0]
        mode2 = stats.mode(vector_2, keepdims=True).mode[0]
        diff = mode1 - mode2
        return mode1, mode2, diff

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
