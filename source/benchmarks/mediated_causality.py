""" Mediated causality (front-door criterion) benchmarks """
import numpy as np
import pandas as pd
from source.utils import create_missing_directory
from source.utils import is_divisible_by_9
from source.utils import check_probability
from source.utils import QuestionBank

def probability_x(arr,var_idx,outcome_idx):
    """ Compute P(x) """
    # can change it to P_y or P_z by changing var_idx
    # var_idx = 0,1,2 for x,y,z
    # outcome_idx = 0 or 1
    filtered_rows = arr[arr[:, var_idx] == outcome_idx]
    return np.sum(filtered_rows[:, 3]) / np.sum(arr[:, 3])

def probability_z_given_x(data, x, z):
    """ Compute P(z|x) """

    # Step 1: Get the values from the fourth column where
    # rows match [x,0,z] and [x,1,z], then sum them
    mask1 = (data[:, 0] == x) & (data[:, 1] == 0) & (data[:, 2] == z)
    mask2 = (data[:, 0] == x) & (data[:, 1] == 1) & (data[:, 2] == z)
    sum_step1 = np.sum(data[mask1, 3]) + np.sum(data[mask2, 3])

    # Step 2: Get all values from the fourth column where the first
    # column equals x, then sum them
    mask3 = (data[:, 0] == x)
    sum_step2 = np.sum(data[mask3, 3])

    return sum_step1 / sum_step2 if sum_step2 != 0 else np.nan

def probability_y_given_x_and_z(data, x, y, z):
    """ Compute P(y|x,z) """
    mask1 = (data[:, 0] == x) & (data[:, 1] == y) & (data[:, 2] == z)
    numerator = data[mask1, 3]
    mask2 = (data[:, 0] == x) & (data[:, 1] == 0) & (data[:, 2] == z)
    mask3 = (data[:, 0] == x) & (data[:, 1] == 1) & (data[:, 2] == z)
    denominator = data[mask2, 3] + data[mask3, 3]
    if denominator != 0:
        return (numerator / denominator)[0]
    return np.nan

def causality_from_table(data, test, n_bootstrap=1000):
    """ Compute P(Y=1|do(X=1))-P(Y=1|do(X=0)) from frequency table"""

    n = np.sum(data[:,3])

    if test in ("tdist", "arithmetic"):

        # Calc P(x), P(z|x), P(Y=1|x,z) to get  P(Y=1|do(X=1))

        # P(x)
        p_x0 = probability_x(data,0,0)
        p_x1 = probability_x(data,0,1)
        check_probability(p_x0)
        check_probability(p_x1)

        # P(z|x) = P(x,z)/P(x)
        p_z0_given_x0 = probability_z_given_x(data, 0, 0)
        check_probability(p_z0_given_x0)
        p_z1_given_x0 = probability_z_given_x(data, 0, 1)
        check_probability(p_z1_given_x0)
        p_z0_given_x1 = probability_z_given_x(data, 1, 0)
        check_probability(p_z0_given_x1)
        p_z1_given_x1 = probability_z_given_x(data, 1, 1)
        check_probability(p_z1_given_x1)

        # Get P(Y=1|x,z) = P(x,Y=1,z) / P(x,z)
        p_y1_given_x0_z0 = probability_y_given_x_and_z(data, 0, 1, 0)
        check_probability(p_y1_given_x0_z0)
        p_y1_given_x0_z1 = probability_y_given_x_and_z(data, 0, 1, 1)
        check_probability(p_y1_given_x0_z1)
        p_y1_given_x1_z0 = probability_y_given_x_and_z(data, 1, 1, 0)
        check_probability(p_y1_given_x1_z0)
        p_y1_given_x1_z1 = probability_y_given_x_and_z(data, 1, 1, 1)
        check_probability(p_y1_given_x1_z1)

        # compute P(Y=1|do(X=1))
        p_y1_do_x1 = (
            p_z0_given_x1 * (p_y1_given_x0_z0 * p_x0 + p_y1_given_x1_z0 * p_x1)
            + p_z1_given_x1 * (p_y1_given_x0_z1 * p_x0 + p_y1_given_x1_z1 * p_x1)
        )
        check_probability(p_y1_do_x1)

        # compute P(Y=1|do(X=0))
        p_y1_do_x0 = (
            p_z0_given_x0 * (p_y1_given_x0_z0 * p_x0 + p_y1_given_x1_z0 * p_x1)
            + p_z1_given_x0 * (p_y1_given_x0_z1 * p_x0 + p_y1_given_x1_z1 * p_x1)
        )
        check_probability(p_y1_do_x0)

        if test == 'arithmetic':
            equiv_str = (
                "Please perform the following calculation and provide the answer: "
                f"{p_z0_given_x1:.2f} × ({p_y1_given_x0_z0:.2f} × {p_x0:.2f} "
                f"+ {p_y1_given_x1_z0:.2f} × {p_x1:.2f}) + "
                f"{p_z1_given_x1:.2f} × ({p_y1_given_x0_z1:.2f} × {p_x0:.2f} "
                f"+ {p_y1_given_x1_z1:.2f} × {p_x1:.2f}) - "
                f"{p_z0_given_x0:.2f} × ({p_y1_given_x0_z0:.2f} × {p_x0:.2f} "
                f"+ {p_y1_given_x1_z0:.2f} × {p_x1:.2f}) + "
                f"{p_z1_given_x0:.2f} × ({p_y1_given_x0_z1:.2f} × {p_x0:.2f} "
                f"+ {p_y1_given_x1_z1:.2f} × {p_x1:.2f})."
            )

            term1 = np.round(p_z0_given_x1, 2) * (
                np.round(p_y1_given_x0_z0, 2) * np.round(p_x0, 2)
                + np.round(p_y1_given_x1_z0, 2) * np.round(p_x1, 2)
            )
            term2 = np.round(p_z1_given_x1, 2) * (
                np.round(p_y1_given_x0_z1, 2) * np.round(p_x0, 2)
                + np.round(p_y1_given_x1_z1, 2) * np.round(p_x1, 2)
            )
            term3 = np.round(p_z0_given_x0, 2) * (
                np.round(p_y1_given_x0_z0, 2) * np.round(p_x0, 2)
                + np.round(p_y1_given_x1_z0, 2) * np.round(p_x1, 2)
            )
            term4 = np.round(p_z1_given_x0, 2) * (
                np.round(p_y1_given_x0_z1, 2) * np.round(p_x0, 2)
                + np.round(p_y1_given_x1_z1, 2) * np.round(p_x1, 2)
            )
            equiv_ans = str(term1 + term2 - (term3 + term4))
            return float(equiv_ans), equiv_str, equiv_ans

        p_diff = p_y1_do_x1 - p_y1_do_x0
        se_p = np.sqrt( p_y1_do_x1*(1-p_y1_do_x1)/n + p_y1_do_x0*(1-p_y1_do_x0)/n )
        p_diff_ci_upper = p_diff + 1.96*se_p
        p_diff_ci_lower = p_diff - 1.96*se_p

        return p_diff,p_diff_ci_lower,p_diff_ci_upper,n

    # if test == 'bootstrap'
    boot_data = [
        {'X': 0, 'Y': 0, 'Z': 0, 'count': data[:,3][0]},
        {'X': 0, 'Y': 0, 'Z': 1, 'count': data[:,3][1]},
        {'X': 0, 'Y': 1, 'Z': 0, 'count': data[:,3][2]},
        {'X': 0, 'Y': 1, 'Z': 1, 'count': data[:,3][3]},
        {'X': 1, 'Y': 0, 'Z': 0, 'count': data[:,3][4]},
        {'X': 1, 'Y': 0, 'Z': 1, 'count': data[:,3][5]},
        {'X': 1, 'Y': 1, 'Z': 0, 'count': data[:,3][6]},
        {'X': 1, 'Y': 1, 'Z': 1, 'count': data[:,3][7]},
        ]
    df_counts = pd.DataFrame(boot_data)
    df_full = (
        df_counts.loc[df_counts.index.repeat(df_counts["count"])]
        .drop(columns="count")
        .reset_index(drop=True)
    )

    boot_diffs = bootstrap_p_y1_do_diff(
        df_full,
        n_boot=n_bootstrap,
    )

    # Point estimate
    p_diff = np.mean(boot_diffs)

    # Standard error
    #se = np.std(boot_diffs)

    # 95% confidence interval
    p_diff_ci_lower, p_diff_ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    return p_diff,p_diff_ci_lower,p_diff_ci_upper,n

def bootstrap_p_y1_do_diff(df, n_boot, seed=42):
    """Bootstrap diff = P(Y=1|do(X=1)) - P(Y=1|do(X=0)) from Pandas dataframe"""
    np.random.seed(seed)
    estimates = []
    n = len(df)

    for _ in range(n_boot):
        # Resample with replacement
        sample = df.sample(n=n, replace=True)

        if (
            (sample['X'] == 1).any()
            and sample.groupby(['X', 'Z'])['Y'].count().min() >= 1
        ):
            # Estimate do(X=1) and do(X=0) from the sample
            try:
                p1 = estimate_p_y1_do_x1_dataframe(sample)
                p0 = estimate_p_y1_do_x0_dataframe(sample)
                estimates.append(p1 - p0)
            except ValueError:
                continue  # Skip sample if it fails (e.g., missing subgroup)

    return np.array(estimates)

def estimate_p_y1_do_x0_dataframe(df):
    """Compute P(Y=1|do(X=0)) from Pandas dataframe"""
    # Estimate P(x')
    p_x = df['X'].value_counts(normalize=True)
    # Estimate P(z | X=0)
    p_z_given_x0 = df[df['X'] == 0]['Z'].value_counts(normalize=True)
    # Estimate P(Y=1 | x', z)
    p_y1_given_xz = df.groupby(['X', 'Z'])['Y'].mean()

    p = 0.0
    for z in p_z_given_x0.index:
        inner_sum = 0.0
        for x in p_x.index:
            if (x, z) in p_y1_given_xz:
                p_y1 = p_y1_given_xz[(x, z)]
                inner_sum += p_y1 * p_x[x]
        p += p_z_given_x0[z] * inner_sum

    return p

def estimate_p_y1_do_x1_dataframe(df):
    """Compute P(Y=1|do(X=1)) from Pandas dataframe"""
    p_x = df['X'].value_counts(normalize=True)
    p_z_given_x1 = df[df['X'] == 1]['Z'].value_counts(normalize=True)
    p_y1_given_xz = df.groupby(['X', 'Z'])['Y'].mean()

    p = 0.0
    for z in p_z_given_x1.index:
        inner_sum = 0.0
        for x in p_x.index:
            if (x, z) in p_y1_given_xz:
                p_y1 = p_y1_given_xz[(x, z)]
                inner_sum += p_y1 * p_x[x]
        p += p_z_given_x1[z] * inner_sum

    return p

def generate_dataset(
    size=8,
    sum_target_range=(0.7, 0.9),
    probability_of_pattern=0.90,
):
    """ Randomly generate data for the frequency table """

    chance = np.random.uniform(0, 1)

    if chance >= probability_of_pattern:
        array = np.random.uniform(0, 1, size)
        array = array / np.sum(array)
    else:
        two_sample_sum = np.random.uniform(sum_target_range[0], sum_target_range[1])
        two_samples = np.random.uniform(0.4, 0.6, 2)
        two_samples = two_samples / np.sum(two_samples) * two_sample_sum

        remaining_samples = np.random.uniform(0, 1, size-2)
        remaining_samples = remaining_samples / np.sum(remaining_samples) * (1.-two_sample_sum)
        array = np.append(two_samples,remaining_samples)
        np.random.shuffle(array)

    return array

def generate_table(xyz, generated_array, factor, number_type):
    """ Generate the frequency table """
    if number_type == 'integers':
        samples = np.round(
            np.transpose(
                np.array([generated_array * factor])
            )
        )
    elif number_type == 'rational numbers':
        samples = np.transpose(
            np.array([generated_array * factor])
        )
    else:
        raise ValueError(
            "Number type for frequency table "
            "not rational nor integer."
        )
    return np.hstack((xyz,samples))

def get_table():
    """ table of binary variables """
    xyz = np.array([[0,0,0],
                    [0,0,1],
                    [0,1,0],
                    [0,1,1],
                    [1,0,0],
                    [1,0,1],
                    [1,1,0],
                    [1,1,1]])
    return xyz

def get_dictionaries(): # SHOULD BE A NEW CLASS. ANSWER COUNTERS. DATA.
    """ dictionary of counters and data for each problem """
    easy = {
        "n_problems": 0, # number of problems at this difficulty
        "n_A": 0, # number of problems at this difficulty with answer A
        "n_B": 0, # number of problems at this difficulty with answer B
        "n_C": 0, # number of problems at this difficulty with answer C
        "n_samples": np.empty((), dtype=object), # total number of easy samples 
        "questions": np.empty((), dtype=object),
        "answers": np.empty((), dtype=object),
        "p_diff": np.empty((), dtype=object),
        "p_diff_ci_upper": np.empty((), dtype=object),
        "p_diff_ci_lower": np.empty((), dtype=object),
        "table": []
    }
    medm = easy.copy()
    hard = easy.copy()
    return easy, medm, hard

def get_names(exam_name):
    """ Variable names for prompt"""
    if exam_name.startswith("MediatedCausalitySmoking"):
        return (
            "smoke", # x_name
            "have lung cancer",  # y_name
            "have tar deposits in lungs", # z_name
            "smoking", # x_name_verb
            "lung cancer", # y_name_noun
        )
    return (
            "X", # x_name
            "Y",  # y_name
            "Z", # z_name
            "doing X", # x_name_verb
            "Y", # y_name_noun
        )

class MediatedCausality():

    def __init__(self, plot_path, exam_name, **kwargs):

        self.plot_path = plot_path
        self.exam_name = exam_name

        # generation parameters:
        self.plot_flag = kwargs.get('plot_flag', False)
        self.generate_flag = kwargs.get('generate_flag', True)
        self.verbose = kwargs.get('verbose', False)        
        self.answer_proportions = kwargs.get(
            "answer_proportions",
            [0.333, 0.333, 0.333], # Ratios of A, B, C correct answers
        )
        # n_samples = number of sample sizes generated per causal example
        self.n_samples = kwargs.get('n_samples', 50)
        self.min_power10_sample_size = kwargs.get(
            'min_power10_sample_size', 
            1
        )
        self.max_power10_sample_size = kwargs.get(
            'max_power10_sample_size', 
            4
        )
        self.difficulty_thresholds = kwargs.get(
            'difficulty_thresholds', 
            np.array([0.05,0.25])
        )
        self.ci_method = (exam_name).split('_')[1]
        self.exam_name_wo_ci_method = (exam_name).split('_')[0]
        self.n_problems = kwargs.get('n_problems', 360)
        self.n_bootstrap = kwargs.get('n_bootstrap', 1000)
        if not is_divisible_by_9(self.n_problems):
            raise ValueError(
                "\n The number of problems specified is not divisible by 9."
                "Benchmark not created."
            )
        (
            self.x_name,
            self.y_name,
            self.z_name,
            self.x_name_verb,
            self.y_name_noun,
        ) = get_names(self.exam_name)

        if self.generate_flag: # necessary for testing
            self.make_problems()

    def make_problems(self):
        """ Generate mediated causality problems """

        #easy, medm, hard = get_dictionaries()
        qb = QuestionBank(target_per_bin=int(self.n_problems/9))
        xyz = get_table()

        test_complete = False
        example_idx = 0
        while not test_complete:

            # generate a causal example:
            factor = np.logspace(
                self.min_power10_sample_size,
                self.max_power10_sample_size,
                num=self.n_samples,
                endpoint=True,
            )
            generated_array = generate_dataset()

            # these range over varied n_samples:
            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)
            difficulty_tmp = np.empty(self.n_samples, dtype=object)               
            p_diff_tmp = np.zeros([self.n_samples])
            p_diff_ci_upper_tmp = np.zeros([self.n_samples])
            p_diff_ci_lower_tmp = np.zeros([self.n_samples])
            n_samples_tmp = np.zeros([self.n_samples])
            causality_tmp = np.zeros([self.n_samples]) # (for plotting)
            table_tmp =  np.zeros([self.n_samples,8,4])

            for i in reversed(range(self.n_samples)):

                table = generate_table(xyz, generated_array, factor[i], 'integers')
                table_tmp[i,:,:] = table

                (
                    p_diff_tmp[i],
                    p_diff_ci_lower_tmp[i],
                    p_diff_ci_upper_tmp[i],
                    n_samples_tmp[i],
                ) = causality_from_table(
                    table, 
                    self.ci_method, 
                    self.n_bootstrap
                )

                # Calculate the difficulty level
                difficulty_tmp[i] = self.assign_difficulty(np.abs(p_diff_tmp[i]))

                # Get questions:
                questions_tmp[i] = self.get_prompts(table)

                # Record the solutions:
                causality_tmp[i], answers_tmp[i] = self.record_solutions(
                    p_diff_ci_lower_tmp[i],
                    p_diff_ci_upper_tmp[i],
                )

            # Randomly select one case from the generated causal examples
            # with different numbers of samples:
            random_choice_of_n_samples = np.random.randint(
                0,
                high=self.n_samples,
                size=self.n_samples
            )

            # Make sure the random choice has a non-NaN p_diff:
            p_diff_is_not_nan = False
            k = 0
            subsample_idx = 0
            while not p_diff_is_not_nan:
                subsample_idx = random_choice_of_n_samples[k]
                if np.isnan(p_diff_tmp[subsample_idx]):
                    k += 1
                else:
                    p_diff_is_not_nan = True

            problem = {
                "question": questions_tmp[subsample_idx],
                "solution": answers_tmp[subsample_idx],
                "difficulty": difficulty_tmp[subsample_idx],            
                "p_diff": p_diff_tmp[subsample_idx],
                "p_diff_ci_lower": p_diff_ci_lower_tmp[subsample_idx],
                "p_diff_ci_upper": p_diff_ci_upper_tmp[subsample_idx],
                "n_samples": n_samples_tmp[subsample_idx],
                "causality": causality_tmp[subsample_idx],                 
                "table": table_tmp[subsample_idx],
                "p_diff_all": p_diff_tmp, # all sample sizes
                "p_diff_ci_lower_all": p_diff_ci_lower_tmp,
                "p_diff_ci_upper_all": p_diff_ci_upper_tmp,
                "n_samples_all": n_samples_tmp, 
                "causality_all": causality_tmp,               
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
                self.make_plot(problem)   

            # Check if ready:
            if qb.is_full():
                final_set = qb.get_balanced_set()
                if self.verbose:
                    print("Test is complete:", len(final_set), "questions")
                test_complete = True
                # Pull attributes from qb to MediatedCausality
                qb.p_diff = np.array([q['metadata']['p_diff'] for q in qb.get_balanced_set()])
                qb.p_diff_ci_lower = np.array([q['metadata']['p_diff_ci_lower'] for q in qb.get_balanced_set()])
                qb.p_diff_ci_upper = np.array([q['metadata']['p_diff_ci_upper'] for q in qb.get_balanced_set()])
                qb.n_samples = np.array([q['metadata']['n_samples'] for q in qb.get_balanced_set()])
                qb.table = np.array([q['metadata']['table'] for q in qb.get_balanced_set()])
                qb.example_idx = np.array([q['metadata']['example_idx'] for q in qb.get_balanced_set()])
                qb.name = np.array([q['metadata']['name'] for q in qb.get_balanced_set()])
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

    def get_prompts(self, table):
        """ Get questions for different tests """
        q = []
        if self.exam_name_wo_ci_method in (
            "MediatedCausalitySmoking",
            "MediatedCausality",
        ):
            q = (
            f"Consider the following causal inference problem. "
            f"The number of samples that do not {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name} is {int(table[0,3]):d}. "
            f"{int(table[1,3]):d} samples do not {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[2,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[3,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[4,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[5,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[6,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[7,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"Does {self.x_name_verb} cause {self.y_name_noun}? Please "
            f"answer 'A' for yes, 'B' for no, or 'C' for uncertain. "
            f"Please use only the data provided here and the 95% confidence "
            f"level."
            )
        elif self.exam_name == 'MediatedCausalityWithMethod_tdist':
            q = (
            f"Consider the following causal inference problem. "
            f"The number of samples that do not {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name} is {int(table[0,3]):d}. "
            f"{int(table[1,3]):d} samples do not {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[2,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[3,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[4,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[5,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[6,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[7,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"Does {self.x_name_verb} cause {self.y_name_noun}? Use the "
            f"front-door criterion to determine if smoking causes cancer "
            f"from the provided data. Use the standard error of proportion "
            f"and t distribution on the final front door probability " 
            f" difference to calculate the 95% confidence level intervals. "
            f"Use the the 95% confidence levels to answer 'A' for yes, "
            f"'B' for no, or 'C' for uncertain. Please use only the data "
            f"provided here."
            )
        elif self.exam_name == 'MediatedCausalityWithMethod_bootstrap':
            q = (
            f"Consider the following causal inference problem. "
            f"Please answer only with 'A', 'B', or 'C'. "
            f"The number of samples that do not {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name} is {int(table[0,3]):d}. "
            f"{int(table[1,3]):d} samples do not {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[2,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[3,3]):d} samples do not {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[4,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[5,3]):d} samples do {self.x_name}, do not "
            f"{self.y_name}, and do {self.z_name}. "
            f"{int(table[6,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do not {self.z_name}. "
            f"{int(table[7,3]):d} samples do {self.x_name}, do "
            f"{self.y_name}, and do {self.z_name}. "
            f"Does {self.x_name_verb} cause {self.y_name_noun}? Use the "
            f"front-door criterion of causal inference and only the provided " 
            f"data. Bootstrap the final front door probability difference to " 
            f"calculate 95% confidence level by numerically estimating " 
            f"cumulative probabilities. Use the the 95% confidence levels to " 
            f"answer 'A' for yes, 'B' for no, or 'C' for uncertain. "
            f"Please use only the data provided here."
            )
        return q

    # pylint: disable=too-many-arguments
    def make_plot(self,problem):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            #create_missing_directory(self.plot_path)
            import matplotlib # pylint: disable=import-outside-toplevel
            matplotlib.use('Agg') # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel
            low_n = np.power(10.,self.min_power10_sample_size)
            high_n = np.power(10.,self.max_power10_sample_size)
            figname = f"{self.plot_path}/example_{problem['example_idx']}.png"
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

    def record_solutions(self,ci_lower,ci_upper):
        """ Determine if answer is A, B, or C """
        if ci_lower > 0.:
            plot_val = 2 # for plotting
            answer = 'A' # X causes Y
        elif ci_upper < 0.:
            plot_val = 1 # for plotting
            answer = 'B' # not X causes Y
        else:
            plot_val = 0 # for plotting
            answer = 'C' # Uncertain
        return plot_val, answer

    def assign_difficulty(self, diff_value):
            if diff_value <= self.difficulty_thresholds[0]:
                difficulty = 'hard'
            elif diff_value <= self.difficulty_thresholds[1]:
                difficulty = 'medium'
            elif diff_value > self.difficulty_thresholds[1]:
                difficulty = 'easy'
            else: # diff_value = NaN
                difficulty = 'N/A'    
            return difficulty

    def print_problems(self):
        """Print to terminal information on all n_problems"""
        nan_flag = 'No NaNs!'
        idx_list = self.reverse_idx
        for i in range(0,self.n_problems):
            idx = idx_list[i]
            print('\n')
            print(' Problem ',i)
            print(' Q = ',self.questions[idx])
            print(' A = ',self.solutions[idx])
            print(' difficulty = ',self.difficulty[idx])
            print(' N_samples = ',int(self.n_samples[idx]))
            print(
                " p_diff, p_diff_ci_upper, p_diff_ci_lower = ",
                self.p_diff[idx],
                self.p_diff_ci_upper[idx],
                self.p_diff_ci_lower[idx]
            )
            if np.isnan(self.p_diff[idx]):
                nan_flag = 'NaNs!'
            if np.isnan(self.p_diff_ci_upper[idx]):
                nan_flag = 'NaNs!'
            if np.isnan(self.p_diff_ci_lower[idx]):
                nan_flag = 'NaNs!'
        print(nan_flag)

    def get_questions(self):
        """Return a vector (of length n_problems) of the questions 
        for each problem"""
        return self.question

    def get_solutions(self):
        """Return a vector (of length n_problems) of the solutions 
        to each problem"""
        return self.solution

    def get_difficulty(self):
        """Return a vector (of length n_problems) of the difficulty 
        of each problem"""
        return self.difficulty

    def get_n_samples(self):
        """Return a vector (of length n_problems) of the number of samples 
        in each problem"""
        return self.n_samples
    
    def get_tables(self):
        """Return an array of the frequency table in each problem"""
        return self.table
    
    def get_p_diff(self):
        """Return a vector (of length n_problems) of the probability 
        difference in each problem"""
        return self.p_diff
               
    def get_p_diff_ci_upper(self):
        """Return a vector (of length n_problems) of the probability 
        difference upper confidence bound in each problem"""
        return self.p_diff_ci_upper

    def get_p_diff_ci_lower(self):
        """Return a vector (of length n_problems) of the probability 
        difference upper confidence bound in each problem"""
        return self.p_diff_ci_lower
    
    def get_example_idx(self):
        """Return a vector (of length n_problems) of the example indices 
        of each problem"""               
        return self.example_idx

    def get_exam_name(self):
        """Return the benchmark name"""               
        return self.name

#===============================================================================

# TEST

# python3 -m source.benchmarks.mediated_causality

# # # exam_name = 'MediatedCausalityWithMethod_tdist'
# #exam_name = 'MediatedCausality_tdist'
# exam_name = 'MediatedCausalitySmoking_tdist'
# # # exam_name = 'MediatedCausalityArithmetic'
# # # exam_name = 'MediatedCausality_bootstrap'
# # # exam_name = 'MediatedCausalitySmoking_bootstrap'
# # # exam_name = 'MediatedCausalityWithMethod_bootstrap'

# #exam_name = 'MediatedCausalityArithmetic'
# plot_path = './figures/'

# if __name__ == "__main__":
#     exam = MediatedCausality(plot_path, exam_name, plot_flag=False, n_problems=9)
#     #exam = MediatedCausalityArithmetic(n_problems=9)
#     exam.print_problems()
