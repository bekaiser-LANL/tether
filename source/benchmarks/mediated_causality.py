""" Mediated causality (front-door criterion) benchmarks """
import numpy as np
import pandas as pd
from source.utils import is_divisible_by_9
from source.utils import check_probability
from source.utils import QuestionBank

def duplicate_tables(table_data):
    """
    Determines if there are any duplicate frequency tables.
    Expects table_data to be of shape (n_problems, 8).
    
    Returns:
        True if duplicates exist, False otherwise.
    """
    n_rows = table_data.shape[0]

    # Convert each row (length 8 array) into a tuple
    slices = [tuple(table_data[i, :]) for i in range(n_rows)]
    
    # Put slices into a set (sets remove duplicates automatically)
    unique_slices = set(slices)
    
    # If number of unique slices is less than total rows, we have duplicates
    return len(unique_slices) != n_rows

def causality_from_frequency(array):
    """ Compute front-door criterion from frequency table """
    n000, n001, n010, n011, n100, n101, n110, n111 = array
    
    # Protect all denominators
    try:
        denom1 = n000 + n010
        denom2 = n100 + n110
        denom3 = n001 + n011
        denom4 = n101 + n111
        n = np.sum(array)
        denom5 = n111 + n101 + n110 + n100
        denom6 = n011 + n001 + n010 + n000

        #if denom1 == 0 or denom2 == 0 or denom3 == 0 or denom4 == 0 or n == 0 or denom5 == 0 or denom6 == 0:
        if any(d == 0 
               for d in [
                   denom1,
                   denom2,
                   denom3,
                   denom4,
                   n,
                   denom5, 
                   denom6
                ]
        ):
            return np.nan  # Safe fallback
        
        A = n010*(n000+n010+n001+n011)/denom1 + n110*(n100+n110+n101+n111)/denom2
        B = n011*(n000+n010+n001+n011)/denom3 + n111*(n100+n110+n101+n111)/denom4
        
        PdoX1 = ((n110+n100)*A + (n111+n101)*B) / (n*denom5)
        PdoX0 = ((n010+n000)*A + (n011+n001)*B) / (n*denom6)
        
        dP = PdoX1 - PdoX0
        return dP
    
    except ZeroDivisionError:
        return np.nan

def generate_dataset_by_difficulty( difficulty, difficulty_threshold, factor_range ):
    """ Generates a single set of sample frequencies """
    sample_frequencies = []
    counter =0
    done=False
    np.random.shuffle(factor_range) 
    n_try = 4
    if difficulty == 'easy':
        dP = 0.
        while dP < difficulty_threshold[1] and not done:
            for i in range(len(factor_range)):
                exponents = np.random.uniform(0., factor_range[i], size=8)
                powers_of_10 = np.power(10, exponents)
                sample_frequencies = np.random.uniform(low=np.nextafter(0, 1), high=1, size=8)*powers_of_10
                sample_frequencies = np.round(np.maximum(sample_frequencies, 1))
                dP = np.abs(causality_from_frequency(sample_frequencies))
                counter += 1
                if counter == int(n_try*len(factor_range)):
                    sample_frequencies = sample_frequencies*np.nan
                    done = True # tried n_try loops, stil not working.
                    break
    elif difficulty == 'hard' and not done:
        dP = difficulty_threshold[0]+1.
        while dP > difficulty_threshold[0]:
            for i in range(len(factor_range)):
                exponents = np.random.uniform(0, factor_range[i], size=8)
                powers_of_10 = np.power(10, exponents)
                sample_frequencies = np.random.uniform(low=np.nextafter(0, 1), high=1, size=8)*powers_of_10
                sample_frequencies = np.round(np.maximum(sample_frequencies, 1))
                dP = np.abs(causality_from_frequency(sample_frequencies))
                counter += 1
                if counter == int(n_try*len(factor_range)):
                    sample_frequencies = sample_frequencies*np.nan
                    done = True # tried n_try loops, stil not working.
                    break
    elif difficulty == 'medium' and not done:
        criteria = True
        dP = 0.
        while criteria:   
            for i in range(len(factor_range)):
                exponents = np.random.uniform(0, factor_range[i], size=8)
                powers_of_10 = np.power(10, exponents)
                sample_frequencies = np.random.uniform(low=np.nextafter(0, 1), high=1, size=8)*powers_of_10
                sample_frequencies = np.round(np.maximum(sample_frequencies, 1))
                dP = np.abs(causality_from_frequency(sample_frequencies))
                if dP >= difficulty_threshold[0] and dP <= difficulty_threshold[1]:
                    criteria = False
                counter += 1
                if counter == int(n_try*len(factor_range)):
                    sample_frequencies = sample_frequencies*np.nan
                    done = True # tried n_try loops, stil not working.
                    break   
    else:
        raise ValueError(f"Invalid difficulty: {difficulty}. Must be 'easy', 'medium', or 'hard'.")
    if np.sum(sample_frequencies) <= 20.0:
        sample_frequencies = sample_frequencies*np.nan
    return sample_frequencies

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
    #df_counts["count"] = df_counts["count"].clip(upper=1e6)
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
    success_counter = 0

    for idx in range(n_boot):
        # Resample with replacement
        sample = df.sample(n=n, replace=True)
        #print(f"Bootstrap {idx}/{n_boot}")

        if (
            (sample['X'] == 1).any()
            and sample.groupby(['X', 'Z'])['Y'].count().min() >= 1
        ):
            # Estimate do(X=1) and do(X=0) from the sample
            try:
                p1 = estimate_p_y1_do_x1_dataframe(sample)
                p0 = estimate_p_y1_do_x0_dataframe(sample)
                estimates.append(p1 - p0)
                success_counter += 1
            except ValueError:
                continue  # Skip sample if it fails (e.g., missing subgroup)

    #print(f"Successful bootstrap samples: {success_counter}/{n_boot}")
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
    sum_target_range=(0.7, 0.9), # try expanding range
    probability_of_pattern=0.9, # try reducing
):
    """ Randomly generate frequencies for the frequency table """

    chance = np.random.uniform(0, 1)

    if chance >= probability_of_pattern:
        array = np.random.uniform(0, 1, size)
        array = array / np.sum(array)
    else:
        # A single float is drawn from the range defined by sum_target_range:
        two_sample_sum = np.random.uniform(sum_target_range[0], sum_target_range[1])
        # Generate 2 random values from (0.3, 0.7)
        two_samples = np.random.uniform(0.4, 0.6, 2) # try expanding range
        # Normalize and scale them to match the target sum
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
        samples = np.maximum(samples, 1)
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
        self.min_power10_sample_size = kwargs.get(
            'min_power10_sample_size', 
            1
        )
        self.max_power10_sample_size = kwargs.get(
            'max_power10_sample_size', 
            4
        )
        self.difficulty_thresholds = kwargs.get( # NEED TO INPUT FROM COMMAND LINE
            'difficulty_thresholds', 
            np.array([0.05,0.25]) # np.array([0.05,0.25]) for tdist
        )
        self.ci_method = (exam_name).split('_')[1]
        # n_samples = number of sample sizes generated per causal example
        if self.ci_method == 'bootstrap':  
            self.n_samples = kwargs.get('n_samples', 400)
        else:
            self.n_samples = kwargs.get('n_samples', 200)    
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

        if self.verbose:
            print(f"\n Generating {self.n_problems} problems")

        if self.generate_flag: # necessary for testing
            self.make_problems()

    def make_problems(self):
        """ Generate mediated causality problems """

        qb = QuestionBank(target_per_bin=int(self.n_problems/9))
        xyz = get_table()

        test_complete = False
        example_idx = 0
        while not test_complete:
            if self.verbose:
                print('\n New problem generated ')  
            #factor_tmp = []
            factor = []

            # # flag to control whether we continue the while loop
            # # (this expedites the random guessing of new problems)
            # skip_to_next = False 

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

            if self.ci_method == 'tdist':
                
                factor = np.logspace(
                    self.min_power10_sample_size,
                    self.max_power10_sample_size,
                    num=self.n_samples,
                    endpoint=True,
                )
                # frequency proportions:
                generated_array = generate_dataset()

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

            elif self.ci_method == 'bootstrap':

                # generates only hard or easy problems where needed:
                hard_A = qb.count()['A']['hard']
                med_A = qb.count()['A']['medium']
                easy_A = qb.count()['A']['easy'] 
                hard_B = qb.count()['B']['hard']
                med_B = qb.count()['B']['medium']
                easy_B = qb.count()['B']['easy']   
                hard_C = qb.count()['C']['hard']
                med_C = qb.count()['C']['medium']                
                easy_C = qb.count()['C']['easy']
                total = easy_A + med_A + hard_A
                total += easy_B + med_B + hard_B
                total += easy_C + med_C + hard_C

                if easy_C < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make C easy ')
                        print(qb.count())
                    low = 0.25 # works, don't change!
                    high = 1.5 # works, don't change!
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'easy',
                        self.difficulty_thresholds,
                        np.linspace(low,high, num=n_samples, endpoint=True)
                    )
                elif easy_A < int(self.n_problems/9) or easy_B < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make A,B easy ')
                        #print(' self.n_samples = ',self.n_samples)
                    low = 2 # works, don't change!
                    high = 4 # works, don't change!
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'easy',
                        self.difficulty_thresholds,
                        np.linspace(low, high, num=n_samples, endpoint=True)
                    )    
                elif med_C < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make C medium ')
                    low = 0.5 # works, don't change!
                    high = 2.5 #2 # works, don't change!
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'medium',
                        self.difficulty_thresholds,
                        np.linspace(low, high, num=n_samples, endpoint=True)
                    )
                elif hard_A < int(self.n_problems/9) or hard_B < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make A,B hard ')
                    low = 3 # works, don't change!
                    high = 4 # works, don't change!
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'hard',
                        self.difficulty_thresholds,
                        np.linspace(low, high, num=n_samples, endpoint=True)
                    )
                elif med_A < int(self.n_problems/9) or med_B < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make A,B medium ')
                    low = 3 # works, don't change!
                    high = 4 # works, don't change!
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'medium',
                        self.difficulty_thresholds,
                        np.linspace(low, high, num=n_samples, endpoint=True)
                    )  
                elif hard_C < int(self.n_problems/9):
                    if self.verbose:
                        print('\n Trying to make C hard ')
                    low = 0.25 #
                    high = 2 # 
                    n_samples = np.random.randint(2, self.n_samples)
                    factored_array = generate_dataset_by_difficulty(
                        'hard',
                        self.difficulty_thresholds,
                        np.linspace(low, high, num=n_samples, endpoint=True)
                    )                       

                if np.isnan(factored_array).all():
                    # Failure to generate a table w/o NaNs:
                    # Immediately go to next while-loop iteration. 
                    continue

                table = generate_table(xyz, factored_array, np.ones([len(factored_array)]), 'integers')

                # verify that the generated table is unique:
                if total >= 1:
                    factored_array_expanded = np.expand_dims(factored_array, axis=0) # (1,8)
                    tables = qb.get_all_tables()[:,:,3]
                    combined = np.concatenate((tables, factored_array_expanded), axis=0)
                    if duplicate_tables(combined):
                        # Immediately go to next while-loop iteration. 
                        if self.verbose:
                            print('\n Found duplicate frequency table: Skipping')
                        continue

                (
                    p_diff,
                    p_diff_ci_lower,
                    p_diff_ci_upper,
                    n_samples,
                ) = causality_from_table(
                    table, 
                    self.ci_method, 
                    self.n_bootstrap
                )

                # Calculate the difficulty level
                difficulty = self.assign_difficulty(np.abs(p_diff))

                # Get questions:
                questions = self.get_prompts(table)

                # Record the solutions:
                causality, answers = self.record_solutions(
                    p_diff_ci_lower,
                    p_diff_ci_upper,
                )

                problem = {
                    "question": questions,
                    "solution": answers,
                    "difficulty": difficulty,            
                    "p_diff": p_diff,
                    "p_diff_ci_lower": p_diff_ci_lower,
                    "p_diff_ci_upper": p_diff_ci_upper,
                    "n_samples": n_samples,
                    "causality": causality,                 
                    "table": table,
                    "p_diff_all": np.nan,
                    "p_diff_ci_lower_all": np.nan,
                    "p_diff_ci_upper_all": np.nan,
                    "n_samples_all": np.nan, 
                    "causality_all": np.nan,               
                    "subsample_idx": np.nan,
                    "example_idx": example_idx,
                    "name": self.exam_name
                }

            if self.verbose:
                print('\n p_diff = ',problem["p_diff"])
                print(' difficulty = ',problem["difficulty"])
                print(' solution = ',problem["solution"])
                print(' table = ',problem["table"])
                if problem["solution"] == 'A':
                    print(' A ans. p_diff_ci_lower = ',problem["p_diff_ci_lower"])
                if problem["solution"] == 'B':
                    print(' B ans. p_diff_ci_upper = ',problem["p_diff_ci_upper"])

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
                qb.p_diff = np.array([
                    q['metadata']['p_diff']
                    for q in qb.get_balanced_set()]
                )
                qb.p_diff_ci_lower = np.array([
                    q['metadata']['p_diff_ci_lower']
                    for q in qb.get_balanced_set()]
                )
                qb.p_diff_ci_upper = np.array([
                    q['metadata']['p_diff_ci_upper']
                    for q in qb.get_balanced_set()]
                )
                qb.n_samples = np.array([
                    q['metadata']['n_samples']
                    for q in qb.get_balanced_set()]
                )
                qb.table = np.array([
                    q['metadata']['table']
                    for q in qb.get_balanced_set()]
                )
                qb.example_idx = np.array([
                    q['metadata']['example_idx']
                    for q in qb.get_balanced_set()]
                )
                qb.name = np.array([q['metadata']['name'] for q in qb.get_balanced_set()])
                qb.solution = [q['solution'] for q in qb.get_balanced_set()]
                qb.question = [q['question'] for q in qb.get_balanced_set()]           
                qb.difficulty = [q['difficulty'] for q in qb.get_balanced_set()]   
                for name, value in qb.__dict__.items():
                    setattr(self, name, value)

            else:
                example_idx += 1 # loop over examples

            if self.verbose:
                print("\n Still building test. Current count:", qb.count())

        tables = qb.get_all_tables()[:,:,3]
        if duplicate_tables(tables):
            print('\n Duplicate tables detected.')
        else:
            print('\n No duplicate tables')    

        print('\n Done! ')

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
