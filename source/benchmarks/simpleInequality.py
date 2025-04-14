import numpy as np
import math as ma
import random
from scipy.stats import t

class simpleInequality():

    def __init__(self, n_numbers = 10, n_problems=18):
        self.n_problems = n_problems #all tests need this
        self.n_numbers = n_numbers
        self.metadata = {
            "Name": 'simpleInequality'
        }
        self.make_problems() #all tests need this

    def make_problems(self): # all tests need this
        self.questions = [] # all tests need this
        self.solutions = [] # all tests need this
        for i in range(0,self.n_problems): # all tests need this

            v1, v2, q_str = self.generate_question()
            self.questions = np.append(self.questions,q_str)

            #ans_str1 = mean#vector_str = "[" + ", ".join('{:.2f}'.format(x) for x in v1) + "]"
            #ans_str2 = mean#vector_str = "[" + ", ".join('{:.2f}'.format(x) for x in v2) + "]"
            label = self.classify_mean_difference(v1,v2)
            print('label=',label)
            self.solutions = np.append(self.solutions,label)


    def generate_question(self): # all tests need this
        v1, v2 = self.generate_dataset()
        #v1 = [v1 for v1, _ in vectors]
        #v2 = [v2 for _, v2 in vectors]

        # Convert the list of numbers to a space-separated string
        v1numbers_str = " ".join(map(str, v1)) 
        v2numbers_str = " ".join(map(str, v2)) 
        
        # Construct the question with image and add prompt
        q_str = f"Vector 1: {v1numbers_str} Vector 2: {v2numbers_str} Which vector has the higher mean? A: Vector 1 B: Vector 2 C: Uncertain Answer with one letter only: A, B, or C. Answer:"
        #Vector 1: {v1numbers_str} Vector 2: {v2numbers_str} A:Vector 1 has the higher mean, B:Vector 2 has the higher mean, C: Uncertain Responde only with a single letter. Do not repeat the prompt. Answer:"

        return v1, v2, q_str

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

    # Generate the dataset
    #vector_sets = generate_dataset()

    # Example: Check a few sets
    #for i in range(3):
    #    v1, v2 = vector_sets[i]
    #    print(f"Set {i+1}: mean1 = {np.mean(v1):.3f}, std1 = {np.std(v1):.3f}, "
    #      f"mean2 = {np.mean(v2):.3f}, std2 = {np.std(v2):.3f}, "
    #      f"mean diff = {abs(np.mean(v1) - np.mean(v2)):.3f}")

    def classify_mean_difference(self, v1, v2, epsilon=0.05):
        mean1 = np.mean(v1)
        mean2 = np.mean(v2)

        if abs(mean1 - mean2) < epsilon:
            return 'C'  # Uncertain
        elif mean1 > mean2:
            return 'A'
        else:
            return 'B'

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
