import numpy as np
import os
import re

NPZ_FILE_PATH_1 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_1/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_2 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_2/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_3 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_3/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_4 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_4/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_5 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_5/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_6 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_6/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_7 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_7/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_8 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_8/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_9 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_9/blank/SimpleInequality_tdist_0.npz" 
NPZ_FILE_PATH_10 = "/Users/l281800/Desktop/SimpleInequality/benchmark_results_10/blank/SimpleInequality_tdist_0.npz" 

def get_vectors(text):
    # Extract using regex
    vec1_str = re.search(r"Vector 1:(.*?)Vector 2:", text, re.DOTALL).group(1).strip()
    vec2_str = re.search(r"Vector 2:(.*?)(?:\n|$)", text, re.DOTALL).group(1).strip()

    # Convert to numpy arrays
    vector1 = np.fromstring(vec1_str, sep=' ')
    vector2 = np.fromstring(vec2_str, sep=' ')

    #print("Vector 1:", vector1)
    #print("Vector 2:", vector2)
    return vector1,vector2


class VectorPairCollector:
    def __init__(self):
        self.vector_pairs = []      # Will be converted to array later
        self.solutions = []         # List of scalars
        self.difficulties = []      # List of scalars

    def load_npz(self, npz_path):
        if not os.path.isfile(npz_path):
            print(f"❌ File not found: {npz_path}")
            return

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # print(f"✅ Keys in '{npz_path}':")
                # for key in data.keys():
                #     print(f" - {key}")

                if not all(k in data for k in ("question", "solution", "difficulty")):
                    print("\n⚠️ Required keys ('question', 'solution', 'difficulty') not all found.")
                    return

                questions = data["question"]
                solutions = data["solution"]
                difficulties = data["difficulty"]

                if not (len(questions) == len(solutions) == len(difficulties)):
                    print("\n⚠️ Length mismatch between 'question', 'solution', and 'difficulty'.")
                    return

                for i, item in enumerate(questions):
                    #print(f"\n [{i}] ")
                    vector_pair = get_vectors(item)
                    #print(np.shape(vector_pair))
                    self.vector_pairs.append(vector_pair)
                    self.solutions.append(solutions[i])
                    self.difficulties.append(difficulties[i])

        except Exception as e:
            print(f"❗ Error reading '{npz_path}': {e}")

    def get_full_data(self):
        """
        Returns:
            dict of NumPy arrays with keys:
                - 'vector_pairs': shape (N, 2) of vector pairs (each a numpy array)
                - 'solutions': shape (N,)
                - 'difficulties': shape (N,)
        """
        return {
            "vector_pairs": np.array(self.vector_pairs, dtype=object),  # shape: (N, 2)
            "solutions": np.array(self.solutions),
            "difficulties": np.array(self.difficulties)
        }

    def filter_by_solution_and_difficulty(self, solution_value="A", difficulty_level="easy"):
        """
        Filters and returns vector pairs, solutions, and difficulties that match the given criteria.

        Args:
            solution_value (str): Desired solution label ('A', 'B', 'C', etc.)
            difficulty_level (str): Desired difficulty level ('easy', 'intermediate', 'difficult')

        Returns:
            dict: {
                'vector_pairs': np.array,
                'solutions': np.array,
                'difficulties': np.array
            }
        """
        all_data = self.get_full_data()

        solutions = all_data["solutions"]
        difficulties = all_data["difficulties"]
        vector_pairs = all_data["vector_pairs"]

        # Boolean mask
        mask = (solutions == solution_value) & (difficulties == difficulty_level)

        return {
            "vector_pairs": vector_pairs[mask],
            "solutions": solutions[mask],
            "difficulties": difficulties[mask]
        }

def get_diff_stats(filtered_data):
    d = filtered_data["vector_pairs"][:,0,:]-filtered_data["vector_pairs"][:,1,:] # n_benchmarks, n_categories
    mu = np.mean(np.mean(d,axis=1),axis=0)
    sig = np.nanstd(d,axis=(0,1))
    return mu,sig

def get_all_stats( collector ):

    filtered_data = collector.filter_by_solution_and_difficulty("A", "easy")
    mu_Aeasy,sig_Aeasy = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("A", "medium")
    mu_Amed,sig_Amed = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("A", "hard")
    mu_Ahard,sig_Ahard = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("B", "easy")
    mu_Beasy,sig_Beasy = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("B", "medium")
    mu_Bmed,sig_Bmed = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("B", "hard")
    mu_Bhard,sig_Bhard = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("C", "easy")
    mu_Ceasy,sig_Ceasy = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("C", "medium")
    mu_Cmed,sig_Cmed = get_diff_stats(filtered_data)

    filtered_data = collector.filter_by_solution_and_difficulty("C", "hard")
    mu_Chard,sig_Chard = get_diff_stats(filtered_data)

    mu = np.array([mu_Aeasy,mu_Amed,mu_Ahard,mu_Beasy,mu_Bmed,mu_Bhard,mu_Ceasy,mu_Cmed,mu_Chard])
    sig = np.array([sig_Aeasy,sig_Amed,sig_Ahard,sig_Beasy,sig_Bmed,sig_Bhard,sig_Ceasy,sig_Cmed,sig_Chard])
    return mu, sig

def max_percent_diff(arr):
    a = arr[:, None]
    b = arr[None, :]
    #mean = (a + b) / 2
    #diff = 100 * np.abs(a - b) / mean
    diff = abs(a - b) / ((a + b) / 2) # Symmetric Relative Difference
    return np.max(diff)

if __name__ == "__main__":

    collector = VectorPairCollector()

    collector.load_npz(NPZ_FILE_PATH_1)
    mu_1,sig_1 = get_all_stats( collector )
    print('\n')
    print(' mu_1 = ',mu_1)
    print(' sig_1 = ',sig_1)

    collector.load_npz(NPZ_FILE_PATH_2)
    mu_2,sig_2 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_3)
    mu_3,sig_3 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_4)
    mu_4,sig_4 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_5)
    mu_5,sig_5 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_6)
    mu_6,sig_6 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_7)
    mu_7,sig_7 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_8)
    mu_8,sig_8 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_9)
    mu_9,sig_9 = get_all_stats( collector )

    collector.load_npz(NPZ_FILE_PATH_10)
    mu_10,sig_10 = get_all_stats( collector )

    # actually mu_i are 9 different values need to get a max percent difference for each
    # A easy med hard 
    # B easy med hard
    # C easy med hard

    N_benchmarks = 10
    N_categories = 9 # fixed
    str_categories = np.array([ 'A easy','A med', 'A hard', 'B easy', 'B med', 'B hard', 'C easy','C med', 'C hard'])
    MU = np.zeros([N_categories,N_benchmarks])
    SIG = np.zeros([N_categories,N_benchmarks])
    print('\n')
    for j in range(0,N_categories):
        MU[j,:] = np.abs(np.array([mu_1[j], mu_2[j], mu_3[j], mu_4[j], mu_5[j], mu_6[j], mu_7[j], mu_8[j], mu_9[j], mu_10[j]]))
        SIG[j,:] = np.abs(np.array([sig_1[j], sig_2[j], sig_3[j], sig_4[j], sig_5[j], sig_6[j], sig_7[j], sig_8[j], sig_9[j], sig_10[j]]))
        print(f" Problem type: {str_categories[j]}\n Max % difference between means for 10 different generated datasets: {max_percent_diff(MU[j,:]):.2f}%,\n Max % difference between std. devs for 10 different generated datasets: {max_percent_diff(SIG[j,:]):.2f}%")
 
    MU = MU + 1.
    SIG = SIG + 1.
    print('\n')


