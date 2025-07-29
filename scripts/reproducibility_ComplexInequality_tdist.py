from scipy.stats import wasserstein_distance
import numpy as np
import os
import re

NPZ_FILE_PATH_1 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_1/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_2 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_2/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_3 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_3/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_4 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_4/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_5 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_5/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_6 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_6/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_7 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_7/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_8 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_8/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_9 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_9/blank/MediatedCausality_tdist_0.npz" 
NPZ_FILE_PATH_10 = "/Users/l281800/Desktop/MediatedCausality/benchmark_results_10/blank/MediatedCausality_tdist_0.npz" 

NPZ_FILES = np.array([NPZ_FILE_PATH_1,
             NPZ_FILE_PATH_2,
             NPZ_FILE_PATH_3,
             NPZ_FILE_PATH_4,
             NPZ_FILE_PATH_5,
             NPZ_FILE_PATH_6,
             NPZ_FILE_PATH_7,
             NPZ_FILE_PATH_8,
             NPZ_FILE_PATH_9,
             NPZ_FILE_PATH_10])

# Then you have 10 benchmarks, by 9 categories, by 8 values. 
# compute the maximum normalized Wasserstein distance 
# between all 10 benchmarks for the 8 values -> one for each category.

N_benchmarks = 10
N_table = 8

def percent_wasserstein(vec1, vec2, method="range"):
    """
    Compute Wasserstein distance between two vectors and express it as a percentage.

    Args:
        vec1, vec2 (array-like): Vectors of equal length.
        method (str): Normalization method: "mean", "abs_mean", or "range"

    Returns:
        float: Wasserstein distance as a percentage.
    """
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    if vec1.shape != vec2.shape:
        raise ValueError("Vectors must be the same shape.")

    wd = wasserstein_distance(vec1, vec2)

    if method == "mean":
        normalizer = 0.5 * (np.mean(vec1) + np.mean(vec2))
    elif method == "abs_mean":
        normalizer = 0.5 * (np.mean(np.abs(vec1)) + np.mean(np.abs(vec2)))
    elif method == "range":
        normalizer = np.max(np.concatenate([vec1, vec2])) - np.min(np.concatenate([vec1, vec2]))
    else:
        raise ValueError("Unknown method. Choose from: 'mean', 'abs_mean', 'range'.")

    if normalizer == 0:
        return 0.0  # Avoid divide-by-zero

    return 100 * wd / normalizer

class BenchmarkLoader:
    def __init__(self, npz_path):
        self.data = np.load(npz_path, allow_pickle=True)
        
        # Expect these keys to exist
        self.table = self.data["table"]
        self.difficulty = self.data["difficulty"]
        self.solution = self.data["solution"]

    def get_last_k_columns(self, k=8, indices=None):
        """
        Returns the last k columns of the table.
        Optionally filtered by row indices.
        """
        if indices is not None:
            return self.table[indices, -k:]
        return self.table[:, -k:]

    def get_last_k_columns_by_difficulty_and_solution(self, k=8):
        """
        Returns a nested dictionary of the last k columns,
        grouped by difficulty and solution.

        Format:
        {
            'easy': {'A': np.array(... shape (n, k)), ...},
            ...
        }
        """
        result = {}
        difficulties = np.unique(self.difficulty)
        solutions = np.unique(self.solution)

        for diff in difficulties:
            result[diff] = {}
            for sol in solutions:
                mask = (self.difficulty == diff) & (self.solution == sol)
                indices = np.where(mask)[0]
                result[diff][sol] = self.get_last_k_columns(k=k, indices=indices)
        return result

def get_normalized_column_vector(grouped, difficulty, solution, column_index=3):
    """
    Extracts and normalizes a column vector from a specific group in the 'grouped' dict.

    Args:
        grouped (dict): Nested dictionary from BenchmarkLoader.get_last_k_columns_by_difficulty_and_solution()
        difficulty (str): Difficulty level, e.g., 'easy'
        solution (str): Solution choice, e.g., 'A'
        row_index (int): Row within the group to access (default 0)
        column_index (int): Column within the row's matrix to normalize (default 3)

    Returns:
        np.ndarray: Normalized vector of shape (8,), or raises error if out of bounds.
    """
    try:
        group_data = grouped[difficulty][solution]
        normalized = np.zeros([np.shape(group_data)[0],np.shape(group_data)[1]])
        for i in range(np.shape(group_data)[0]):
            vector = group_data[i, :, column_index]
            normalized[i,:] = vector / np.sum(vector)
        return np.mean(normalized,axis=0), np.std(normalized,axis=0)
    except KeyError as e:
        raise ValueError(f"Invalid difficulty or solution key: {e}")
    except IndexError as e:
        raise ValueError(f"Invalid row or column index: {e}")

def get_all_stats_per_benchmark( NPZ_FILE_PATH , difficulty, solution ):

    loader = BenchmarkLoader(NPZ_FILE_PATH)

    # Get last 8 columns grouped by difficulty and solution
    grouped = loader.get_last_k_columns_by_difficulty_and_solution(k=8)

    mu,sig=get_normalized_column_vector(grouped, difficulty, solution)
    #print(mu)
    #print(sig)
    return mu,sig     

def compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, difficulty, solution ):

    N_benchmarks = len(NPZ_FILES)
    MU = np.zeros([N_table,N_benchmarks])
    SIG = np.zeros([N_table,N_benchmarks])
    for i in range(N_benchmarks):
        MU[:,i],SIG[:,i] = get_all_stats_per_benchmark( NPZ_FILES[i], difficulty,  solution)

    tmp = np.array([])
    tmp2 = np.array([])
    for j in range(0,N_benchmarks):
        for i in range(0,N_benchmarks):
            if j != i:
                tmp = np.append(tmp, percent_wasserstein(MU[:,i], MU[:,j]))
                tmp2 = np.append(tmp2, percent_wasserstein(SIG[:,i], SIG[:,j]))

    return np.amax(tmp), np.amax(tmp2),



mu_Aeasy, sig_Aeasy = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "easy", "A" )
print('\n A easy mu and sig: ',mu_Aeasy, sig_Aeasy)

mu_Amed, sig_Amed = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "medium", "A" )
print(' A med mu and sig: ',mu_Amed, sig_Amed)

mu_Ahard, sig_Ahard = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "hard", "A" )
print(' A hard mu and sig: ',mu_Ahard, sig_Ahard)

mu_Beasy, sig_Beasy = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "easy", "B" )
print(' B easy mu and sig: ',mu_Beasy, sig_Beasy)

mu_Bmed, sig_Bmed = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "medium", "B" )
print(' B med mu and sig: ',mu_Bmed, sig_Bmed)

mu_Bhard, sig_Bhard = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "hard", "B" )
print(' B hard mu and sig: ',mu_Bhard, sig_Bhard)

mu_Ceasy, sig_Ceasy = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "easy", "C" )
print(' C easy mu and sig: ',mu_Ceasy, sig_Ceasy)

mu_Cmed, sig_Cmed = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "medium", "C" )
print(' C medium mu and sig: ',mu_Cmed, sig_Cmed)

mu_Chard, sig_Chard = compute_wasserstein_dist_by_difficulty_and_solution( NPZ_FILES, "hard", "C" )
print(' C hard mu and sig: ',mu_Chard, sig_Chard)
