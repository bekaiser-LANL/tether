""" Mediated causality (front-door criterion) benchmarks """
import numpy as np
import pandas as pd
from source.utils import create_missing_directory
from source.utils import is_divisible_by_9
from source.utils import is_divisible_by_3
from source.uncertainty_quantification import check_probability
from source.sorter import Sorter

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

def get_dictionaries():
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

class MediatedCausalityArithmetic():
    """ Prompts for the equivalent arithmetic """

    def __init__(self, **kwargs):

        self.exam_name = 'MediatedCausalityArithmetic'
        self.n_problems = kwargs.get('n_problems', 120)

        # generation parameters:
        self.generate_flag = kwargs.get('generate_flag', True)
        self.answer_proportions = kwargs.get(
            'answer_proportions', 
            [0.333,0.333,0.333]
        ) # ratios of A,B,C correct answers
        self.n_samples = kwargs.get('n_samples', 50)
        # n_samples = number of possible sample sizes per causal example
        self.min_power10_sample_size = kwargs.get('min_power10_sample_size', 1)
        self.max_power10_sample_size = kwargs.get('max_power10_sample_size', 4)
        self.difficulty_thresholds = kwargs.get(
            'difficulty_thresholds',
            np.array([0.05,0.25])
        )
        self.n_problems = kwargs.get('n_problems', 120)
        if not is_divisible_by_3(self.n_problems):
            raise ValueError(
                "\n The number of problems specified is not divisible by 3. "
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

        easy, medm, hard = get_dictionaries()
        xyz = get_table()

        p_diff = []
        difficulty = []
        questions = []
        answers = []
        table_tmp =  np.zeros([self.n_samples,8,4])

        j = 0
        while int(
            easy["n_problems"] + medm["n_problems"] + hard["n_problems"]
        ) < int(self.n_problems):

            sorter = Sorter(self.difficulty_thresholds,self.n_problems)
            abs_p_diff, diff_flag, continue_flag = sorter.initialize()

            # generate a causal scenario
            factor = np.logspace(
                self.min_power10_sample_size,
                self.max_power10_sample_size,
                num=self.n_samples,
                endpoint=True,
            )
            generated_array = generate_dataset()

            for i in reversed(range(self.n_samples)):

                table = generate_table(xyz, generated_array, factor[i], 'integers')
                table_tmp[i,:,:] = table

                (
                    p_diff_tmp,
                    equiv_str,
                    equiv_ans
                ) = causality_from_table(table, 'arithmetic')

                # Calculate the difficulty level
                abs_p_diff = np.abs(p_diff_tmp)
                if np.isnan(p_diff_tmp):
                    # skip to next example
                    continue
                diff_flag = sorter.update_difficulty(abs_p_diff)

                # Check if there are already enough problems
                # generated for that difficulty level
                if sorter.no_more_hard_problems_needed(hard):
                    continue
                if sorter.no_more_medm_problems_needed(medm):
                    continue
                if sorter.no_more_easy_problems_needed(easy):
                    continue

                p_diff = np.append(p_diff,p_diff_tmp)
                difficulty = np.append(difficulty,diff_flag)
                questions = np.append(questions,equiv_str)
                answers = np.append(answers,equiv_ans)
                if sorter.get_diff_flag() == 'hard':
                    hard["n_problems"] += 1
                elif sorter.get_diff_flag() == 'medm':
                    medm["n_problems"] += 1
                elif sorter.get_diff_flag() == 'easy':
                    easy["n_problems"] += 1
                # move on to the next example:
                break

            total = int(
                easy["n_problems"] + medm["n_problems"] + hard["n_problems"]
            )
            print(
                " easy, intermediate, difficult problems =",
                int(easy["n_problems"]),
                int(medm["n_problems"]),
                int(hard["n_problems"])
            )
            print("\n sum of easy, intermediate, difficult problems =", total)
            print(' target total number of problems = ',  int(self.n_problems))

            j += 1 # loop over examples

        self.questions = questions
        self.p_diff = p_diff
        self.difficulty = difficulty
        self.solutions  = answers

        self.metadata = {
            "name": self.exam_name,
            "p_diff": self.p_diff,
            "difficulty": self.difficulty,
            "a_count": np.count_nonzero(self.solutions == 'A'),
            "b_count": np.count_nonzero(self.solutions == 'B'),
            "c_count": np.count_nonzero(self.solutions == 'C'),
            "easy_count": np.count_nonzero(self.difficulty == 'easy'),
            "intermediate_count": np.count_nonzero(self.difficulty == 'intermediate'),
            "difficult_count": np.count_nonzero(self.difficulty == 'difficult'),
            "n_problems": self.n_problems
        }

        print(' Done! ')

    def print_problems(self):
        """ Print to terminal information on all n_problems """
        nan_flag = 'No NaNs!'
        for i in range(0,self.n_problems):
            print('\n')
            print(' Problem ',i)
            print(' Question = ',self.questions[i])
            print(' Answer = ',self.solutions[i])
            print(' p_diff  = ',self.p_diff[i])
            print(' difficulty = ',self.difficulty[i])
            #print(" p_diff = ",self.p_diff[idx])
            if np.isnan(self.p_diff[i]):
                nan_flag = 'NaNs!'
        print('\n ',nan_flag)

    def get_questions(self):
        """Return a vector (of length n_problems) of the questions for each problem"""
        return self.questions

    def get_solutions(self):
        """Return a vector (of length n_problems) of the solutions to each problem"""
        return self.solutions

    def get_metadata(self):
        """Return a dictionary of auxillary benchmark information"""
        return self.metadata

    def get_difficulty(self):
        """Return a vector (of length n_problems) of the difficulty of each problem"""
        return self.difficulty

class MediatedCausality():

    def __init__(self, plot_path, exam_name, **kwargs):

        self.plot_path = plot_path
        self.exam_name = exam_name

        # generation parameters:
        self.plot_flag = kwargs.get('plot_flag', False)
        self.generate_flag = kwargs.get('generate_flag', True)
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

        easy, medm, hard = get_dictionaries()
        xyz = get_table()

        j = 0
        while int(
            easy["n_problems"] + medm["n_problems"] + hard["n_problems"]
        ) < int(self.n_problems):

            sorter = Sorter(self.difficulty_thresholds,self.n_problems)
            abs_p_diff, diff_flag, continue_flag = sorter.initialize()

            # generate a causal scenario:
            factor = np.logspace(
                self.min_power10_sample_size,
                self.max_power10_sample_size,
                num=self.n_samples,
                endpoint=True,
            )
            generated_array = generate_dataset()

            # these range over varied n_samples:
            p_diff_tmp = np.zeros([self.n_samples])
            p_diff_ci_upper_tmp = np.zeros([self.n_samples])
            p_diff_ci_lower_tmp = np.zeros([self.n_samples])
            questions_tmp = np.zeros([self.n_samples],dtype=object)
            answers_tmp = np.zeros([self.n_samples],dtype=object)
            n_samples_tmp = np.zeros([self.n_samples])
            causality_tmp =  np.zeros([self.n_samples]) # (for plotting)
            table_tmp =  np.zeros([self.n_samples,8,4])

            for i in reversed(range(self.n_samples)):

                table = generate_table(xyz, generated_array, factor[i], 'integers')
                table_tmp[i,:,:] = table

                (
                    p_diff_tmp[i],
                    p_diff_ci_lower_tmp[i],
                    p_diff_ci_upper_tmp[i],
                    n_samples_tmp[i],
                ) = causality_from_table(table, self.ci_method, self.n_bootstrap)

                # Calculate the difficulty level
                abs_p_diff = np.abs(p_diff_tmp[i])
                diff_flag = sorter.update_difficulty(abs_p_diff)

                # Check if there are already enough problems
                # generated for that difficulty level
                if sorter.no_more_hard_problems_needed(hard):
                    continue
                elif sorter.no_more_medm_problems_needed(medm):
                    continue
                elif sorter.no_more_easy_problems_needed(easy):
                    continue

                # Get questions:
                questions_tmp[i] = self.get_prompts(table)

                # Record the solutions:
                causality_tmp[i], answers_tmp[i] = self.record_solutions(
                    p_diff_ci_lower_tmp[i],
                    p_diff_ci_upper_tmp[i],
                )

            # Randomly select a total sample size for the generated causal example
            random_choice_of_n_samples = np.random.randint(
                0,
                high=self.n_samples,
                size=self.n_samples
            )
            valid_idx = False
            k=0 # loop over sample sizes
            while not valid_idx:

                select_idx = random_choice_of_n_samples[k]

                variables = {"p_diff_tmp": p_diff_tmp,
                        "p_diff_ci_lower_tmp": p_diff_ci_lower_tmp,
                        "p_diff_ci_upper_tmp": p_diff_ci_upper_tmp,
                        "answers_tmp": answers_tmp,
                        "questions_tmp": questions_tmp,
                        "n_samples_tmp": n_samples_tmp,
                        "table_tmp": table_tmp
                }

                if sorter.get_diff_flag() == 'hard':
                    valid_idx = self.update_dict(
                        hard,
                        variables,
                        select_idx,
                        valid_idx
                    )
                elif sorter.get_diff_flag() == 'medm':
                    valid_idx = self.update_dict(
                        medm,
                        variables,
                        select_idx,
                        valid_idx
                    )
                elif sorter.get_diff_flag() == 'easy':
                    valid_idx = self.update_dict(
                        easy,
                        variables,
                        select_idx,
                        valid_idx
                    )
    
                k += 1 # loop over sample sizes

                if k == int(self.n_samples):
                    # no data available for this causal scenario
                    continue_flag = True
                    # continue causal scenario while loop
                    valid_idx = True
                    # break the sample size selection while loop

            if continue_flag:
                # generate another causal scenario; no data available for
                # this causal scenario
                continue

            self.make_plot(
                j,
                n_samples_tmp,
                p_diff_tmp,
                p_diff_ci_lower_tmp,
                p_diff_ci_upper_tmp,
                select_idx,
                causality_tmp,
                diff_flag,
            )

            total = int(
                easy["n_problems"] + medm["n_problems"] + hard["n_problems"]
            )
            print(
                " easy, intermediate, difficult problems =",
                int(easy["n_problems"]),
                int(medm["n_problems"]),
                int(hard["n_problems"])
            )
            print("\n sum of easy, intermediate, difficult problems =", total)
            print(' target total number of problems = ',  int(self.n_problems))

            j += 1 # loop over examples

        # remove nones, add hard/medium/easy difficulty labels, combine all
        questions = np.concatenate([
            easy["questions"][1:],
            medm["questions"][1:],
            hard["questions"][1:],
        ])
        answers = np.concatenate([
            easy["answers"][1:],
            medm["answers"][1:],
            hard["answers"][1:]
        ])
        p_diff = np.concatenate([
            easy["p_diff"][1:],
            medm["p_diff"][1:],
            hard["p_diff"][1:]
        ])
        p_diff_ci_upper = np.concatenate([
            easy["p_diff_ci_upper"][1:],
            medm["p_diff_ci_upper"][1:],
            hard["p_diff_ci_upper"][1:]
        ])
        p_diff_ci_lower = np.concatenate([
            easy["p_diff_ci_lower"][1:],
            medm["p_diff_ci_lower"][1:],
            hard["p_diff_ci_lower"][1:]
        ])
        n_samples = np.concatenate([
            easy["n_samples"][1:],
            medm["n_samples"][1:],
            hard["n_samples"][1:]
        ])
        difficulty = np.empty(
            easy["n_problems"] + medm["n_problems"] + hard["n_problems"],
            dtype=object
        )
        difficulty[:easy["n_problems"]] = 'easy'
        difficulty[
            easy["n_problems"] : easy["n_problems"] + medm["n_problems"]
        ] = "intermediate"
        difficulty[-hard["n_problems"]:] = 'difficult'

        # randomly shuffle the problem order
        idx = np.random.permutation(self.n_problems)
        self.questions = questions[idx]
        self.solutions  = answers[idx]
        self.difficulty  = difficulty[idx]
        self.n_samples  = n_samples[idx]
        self.p_diff  = p_diff[idx]
        self.p_diff_ci_upper  = p_diff_ci_upper[idx]
        self.p_diff_ci_lower  = p_diff_ci_lower[idx]

        # Ensures order of printed problems match the plot order
        self.reverse_idx = np.argsort(idx)

        self.metadata = {
            "name": self.exam_name,
            "p_diff": self.p_diff,
            "p_diff_ci_upper": self.p_diff_ci_upper,
            "p_diff_ci_lower": self.p_diff_ci_lower,
            "ci_method": self.ci_method,
            "n_samples": self.n_samples,
            "difficulty": self.difficulty,
            "a_count": np.count_nonzero(self.solutions == 'A'),
            "b_count": np.count_nonzero(self.solutions == 'B'),
            "c_count": np.count_nonzero(self.solutions == 'C'),
            "easy_count": np.count_nonzero(self.difficulty == 'easy'),
            "intermediate_count": np.count_nonzero(self.difficulty == 'intermediate'),
            "difficult_count": np.count_nonzero(self.difficulty == 'difficult'),
            "n_problems": self.n_problems
        }

        print(' Done! ')

    def get_prompts(self, table):
        """ Get questions for different tests """
        q = []
        if self.exam_name_wo_ci_method in (
            "MediatedCausalitySmoking",
            "MediatedCausality",
        ):
            q = (
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
            f"answer 'A' for yes, 'B' for no, or 'C' for uncertain."
            )
        elif self.exam_name == 'MediatedCausalityWithMethod_tdist':
            q = (f"The number of samples that do not {self.x_name}, do not "
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
            f"'B' for no, or 'C' for uncertain."
            )
        elif self.exam_name == 'MediatedCausalityWithMethod_bootstrap':
            q = (f"Please answer only with 'A', 'B', or 'C'. "
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
            f"answer 'A' for yes, 'B' for no, or 'C' for uncertain."
            )
        return q

    # pylint: disable=too-many-arguments
    def make_plot(
        self,
        j,
        n_samples_tmp,
        p_diff_tmp,
        p_diff_ci_lower_tmp,
        p_diff_ci_upper_tmp,
        select_idx,
        causality_tmp,
        diff_flag,
    ):
        """ Plot the causal example for varied n_samples """
        if self.plot_flag: # make a plot of the 95% confidence interval
            create_missing_directory(self.plot_path)
            import matplotlib # pylint: disable=import-outside-toplevel
            matplotlib.use('Agg') # pylint: disable=import-outside-toplevel
            import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel
            low_n = np.power(10.,self.min_power10_sample_size)
            high_n = np.power(10.,self.max_power10_sample_size)
            figname = f"{self.plot_path}{self.ci_method}_example_{j}.png"
            fig = plt.figure(figsize=(12, 5))
            ax1=plt.subplot(1,2,1)
            plt.fill_between(
                n_samples_tmp,
                p_diff_ci_lower_tmp,
                p_diff_ci_upper_tmp,
                color="royalblue",
                alpha=0.2,
                label="95% CI"
            )
            plt.plot(n_samples_tmp,p_diff_tmp,color='royalblue',linewidth=1)
            plt.plot(
                n_samples_tmp[select_idx],
                p_diff_tmp[select_idx],
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
                n_samples_tmp,
                causality_tmp,
                color='black',
                linestyle='None',
                marker='o',
                markersize=10,
                linewidth=2
            )
            plt.plot(
                n_samples_tmp[select_idx],
                causality_tmp[select_idx],
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
            plt.title(diff_flag)
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

    def update_dict(self, info, variables, idx, valid_idx):
        """ Update easy, medm, and hard dictionaries """

        if variables["answers_tmp"][idx] == 'A':
            if (
                info["n_A"] < int(self.n_problems / 9)
                and not np.isnan(variables["p_diff_tmp"][idx])
            ):
                info["n_A"] += 1
                info["p_diff"] = np.append(info["p_diff"],variables["p_diff_tmp"][idx])
                info["p_diff_ci_upper"] = np.append(
                    info["p_diff_ci_upper"],
                    variables["p_diff_ci_upper_tmp"][idx]
                )
                info["p_diff_ci_upper"] = np.append(
                    info["p_diff_ci_upper"],
                    variables["p_diff_ci_upper_tmp"][idx],
                )
                info["p_diff_ci_lower"] = np.append(
                    info["p_diff_ci_lower"],
                    variables["p_diff_ci_lower_tmp"][idx]
                )
                info["n_problems"] += 1
                info["questions"] = np.append(info["questions"],variables["questions_tmp"][idx])
                info["answers"] = np.append(info["answers"],variables["answers_tmp"][idx])
                info["n_samples"] = np.append(info["n_samples"],variables["n_samples_tmp"][idx])
                info["table"].append(variables["table_tmp"][idx,:,:])
                valid_idx = True
            else:
                pass

        elif variables["answers_tmp"][idx] == 'B':
            #if info["n_B"] < int(self.n_problems/9) and not np.isnan(variables["p_diff_tmp"][idx]):
            if (
                info["n_B"] < int(self.n_problems / 9)
                and not np.isnan(variables["p_diff_tmp"][idx])
            ):
                info["n_B"] += 1
                info["p_diff"] = np.append(info["p_diff"],variables["p_diff_tmp"][idx])
                info["p_diff_ci_upper"] = np.append(
                    info["p_diff_ci_upper"],
                    variables["p_diff_ci_upper_tmp"][idx]
                )
                info["p_diff_ci_lower"] = np.append(
                    info["p_diff_ci_lower"],
                    variables["p_diff_ci_lower_tmp"][idx]
                )
                info["n_problems"] += 1
                info["questions"] = np.append(info["questions"],variables["questions_tmp"][idx])
                info["answers"] = np.append(info["answers"],variables["answers_tmp"][idx])
                info["n_samples"] = np.append(info["n_samples"],variables["n_samples_tmp"][idx])
                info["table"].append(variables["table_tmp"][idx,:,:])
                valid_idx = True
            else:
                pass

        elif variables["answers_tmp"][idx] == 'C':
            if (
                info["n_C"] < int(self.n_problems / 9)
                and not np.isnan(variables["p_diff_tmp"][idx])
            ):
                info["n_C"] += 1
                info["p_diff"] = np.append(info["p_diff"],variables["p_diff_tmp"][idx])
                info["p_diff_ci_upper"] = np.append(
                    info["p_diff_ci_upper"],
                    variables["p_diff_ci_upper_tmp"][idx]
                )
                info["p_diff_ci_lower"] = np.append(
                    info["p_diff_ci_lower"],
                    variables["p_diff_ci_lower_tmp"][idx]
                )
                info["n_problems"] += 1
                info["questions"] = np.append(info["questions"],variables["questions_tmp"][idx])
                info["answers"] = np.append(info["answers"],variables["answers_tmp"][idx])
                info["n_samples"] = np.append(info["n_samples"],variables["n_samples_tmp"][idx])
                info["table"].append(variables["table_tmp"][idx,:,:])
                valid_idx = True
            else:
                pass

        return valid_idx

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
        """Return a vector (of length n_problems) of the questions for each problem"""
        return self.questions

    def get_solutions(self):
        """Return a vector (of length n_problems) of the solutions to each problem"""
        return self.solutions

    def get_metadata(self):
        """Return a dictionary of auxillary benchmark information"""
        return self.metadata

    def get_difficulty(self):
        """Return a vector (of length n_problems) of the difficulty of each problem"""
        return self.difficulty

    def get_n_samples(self):
        """Return a vector (of length n_problems) of the number of samples in each problem"""
        return self.n_samples

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
