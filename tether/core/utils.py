""" General purpose benchmark functions & classes """
import argparse
import os

import numpy as np

data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")


def get_model_and_indices(string):
    """Split up the filename of a completed benchmark"""
    parts = string.split("_")
    return parts


def detect_duplicate_tables(table_data):
    n_rows = np.shape(table_data)[0]
    slices = [tuple(table_data[i, :, 3]) for i in range(n_rows)]
    seen = set()
    duplicate_pairs = 0
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            if slices[i] == slices[j]:
                duplicate_pairs += 1
                seen.add(i)
                seen.add(j)
    has_duplicates = duplicate_pairs > 0
    return has_duplicates, duplicate_pairs, n_rows


def get_parser(script):
    parser = argparse.ArgumentParser(description="Parse terminal input for tether")
    parser.add_argument(
        "path",
        nargs="?",
        default=data_path,
        help=f"Directory path to /benchmarks/ (default: {data_path})",
    )
    parser.add_argument("exam_name", help="Name of the benchmark")
    if script == "run":
        parser.add_argument("model", help="Name of LLM to use")
        parser.add_argument(
            "--model_path", type=str, help="Optional path for locally downloaded model"
        )
    if script == "analyze":
        parser.add_argument(
            "--grader_model",
            help="Optional: model to use for grading (e.g. gpt-4, claude-3)"
        )
        parser.add_argument(
            "--human_review",
            action="store_true",
            help="Review and correct grades interactively",
        )
        parser.add_argument(
            "--grade_estimate",
            action="store_true",
            help="Analyzer will estimate the grade",
        )
        parser.add_argument(
            "--print_vars",
            action="store_true",
            help="Analyzer will print all variable keys",
        )
        parser.add_argument(
            "--print_responses",
            action="store_true",
            help="Analyzer will print the completed benchmark",
        )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=180,
        help="Number of problems to generate for the benchmark",
    )
    parser.add_argument("--make_plots", action="store_true", help="Enable plotting")
    parser.add_argument(
        "--n_numbers",
        type=int,
        default=10,
        help="Number of integers for standard deviation benchmark",
    )
    parser.add_argument(
        "--exam_idx",
        type=int,
        default=0,
        help="Index for multiple benchmarks of the same type",
    )
    parser.add_argument("--verbose", action="store_true", help="Print to terminal")
    parser.add_argument(
        "--agent", action="store_true", help="Invokes agent to run and analyze code"
    )
    return parser


def strip_after_second_underscore(s):
    """Get the exam_name if it has exam_idx on the end.
    For names like MediatedCausality_tdist_0 it will
    extract the exam_name MediatedCausality_tdist"""
    parts = s.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return s


def get_after_second_underscore(s):
    """Get the exam_idx if the exam_name has one.
    For names like MediatedCausality_tdist_0 it will
    extract the exam_name MediatedCausality_tdist"""
    parts = s.split("_")
    if len(parts) > 2:
        return "_".join(parts[2:])
    return ""


def get_npz_filename_no_model(save_npz_path, exam_name, exam_idx):
    """Determine filename for blank benchmarks"""
    if exam_idx != "unset":
        filename = f"{exam_name}_{exam_idx}.npz"
    else:
        filename = f"{exam_name}.npz"
    npz_filename = os.path.join(save_npz_path, filename)
    return npz_filename

def get_npz_filename(save_npz_path, exam_name, exam_idx, model, agent=False):
    """ Determine filename for completed benchmarks """
    # Replace unsafe filename characters (like colon)
    safe_model = model.replace(":", "-")
    if exam_idx != "unset":
        filename = f"{exam_name}_{safe_model}_{exam_idx}.npz"
        if agent:
            filename = f"{exam_name}_{safe_model}_agent_{exam_idx}.npz"
    else:
        filename = f"{exam_name}_{safe_model}.npz"
        if agent:
            filename = f"{exam_name}_{safe_model}_agent.npz"
    npz_filename = os.path.join(
        save_npz_path,
        filename
    )
    return npz_filename

def get_json_filename(save_json_path, exam_name, exam_idx, model, agent=False):
    """Determine filename for saved JSON benchmark output"""
    # Replace unsafe filename characters (like colon)
    safe_model = model.replace(":", "-")

    if exam_idx != "unset":
        filename = f"{exam_name}_{safe_model}_{exam_idx}.json"
        if agent:
            filename = f"{exam_name}_{safe_model}_agent_{exam_idx}.json"
    else:
        filename = f"{exam_name}_{safe_model}.json"
        if agent:
            filename = f"{exam_name}_{safe_model}_agent.json"
    json_filename = os.path.join(
         save_json_path,
         filename
    )
    return json_filename


def load_saved_benchmark(benchmark_path, exam_name, exam_idx):
    """read saved .npz file containing dict (benchmark)"""
    if exam_idx != "unset":
        filename = os.path.join(benchmark_path, f"{exam_name}_{exam_idx}.npz")
    else:
        filename = os.path.join(benchmark_path, f"{exam_name}.npz")
    data = np.load(filename, allow_pickle=True)
    return data


def is_integer(value):
    """Is value an integer"""
    return isinstance(value, int)


def get_95_CI_tdist(proportion, n_samples):
    """t distribution to estimate standard error"""
    se = standard_error_for_proportion(proportion, n_samples)
    return proportion + 1.96 * se, proportion - 1.96 * se


def standard_error_for_proportion(P, N):
    """Standard error for proportion (probabilities from frequency tabel)"""
    # Reference:
    # Brayer, Edward F. "Calculating the standard error of a proportion."
    # Journal of the Royal Statistical Society Series C:
    # Applied Statistics 6.1 (1957): 67-68.
    return np.sqrt((P * (1.0 - P)) / N)


def check_probability(probability):
    """Verify probability range"""
    if probability > 1.0:
        print("\n ERROR: Probability > 1")
    elif probability < 0.0:
        print("\n ERROR: Probability < 0")
    return


def enforce_probability_bounds(var):
    """Enforce probability range"""
    if var > 1.0:
        var = 1.0
    elif var < 0.0:
        var = 0.0
    return var


def create_missing_directory(directory_path):
    """Checks if a directory exists and makes it if not"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)


def is_divisible_by_9(number):
    """Checks if divisible by 9"""
    return number % 9 == 0


def is_divisible_by_3(number):
    """Checks if divisible by 3"""
    return number % 3 == 0


class QuestionBank:
    """Saves and counts questions/solutions for A,B,C multiple choice"""

    def __init__(self, target_per_bin=1):
        """
        Initialize a tracker for question generation.

        Parameters:
            target_per_bin (int): How many questions
            you want per (choice, difficulty) bin.
        """
        self.target_per_bin = target_per_bin

        self.data = {
            choice: {difficulty: [] for difficulty in ["easy", "medium", "hard"]}
            for choice in ["A", "B", "C"]
        }

    def add_question(self, question_text, correct_choice, difficulty, metadata=None):
        """
        Adds a question to the appropriate bin.
        """
        print(correct_choice)
        if correct_choice not in self.data:
            raise ValueError("Invalid correct choice. Must be 'A', 'B', or 'C'")
        if difficulty not in self.data[correct_choice]:
            raise ValueError("Invalid difficulty. Must be 'easy', 'medium', or 'hard'")

        bin_list = self.data[correct_choice][difficulty]
        if len(bin_list) >= self.target_per_bin:
            # Skip adding to overfilled bin
            return False  # Optionally indicate it was rejected

        entry = {
            "question": question_text,
            "solution": correct_choice,
            "difficulty": difficulty,
            "metadata": metadata or {},
        }
        bin_list.append(entry)
        return True

    def get_all_tables(self):
        """
        Extract all 'table' entries if available,
        fallback to searching through self.data if needed.
        """
        if hasattr(self, "table"):
            return self.table
        else:
            # fallback: try to extract from data structure
            tables = []
            for choice in self.data:
                for difficulty in self.data[choice]:
                    for entry in self.data[choice][difficulty]:
                        metadata = entry.get("metadata", {})
                        if "table" in metadata:
                            tables.append(metadata["table"])
            if tables:
                return np.array(tables)
            raise AttributeError("No tables found in metadata either.")

    def count(self):
        """
        Returns a nested count of how many questions are in each bin.
        """
        return {
            choice: {
                difficulty: len(self.data[choice][difficulty])
                for difficulty in self.data[choice]
            }
            for choice in self.data
        }

    def is_full(self):
        """
        Checks whether all bins are full based on the target count.
        """
        for choice in self.data:
            for difficulty in self.data[choice]:
                if len(self.data[choice][difficulty]) < self.target_per_bin:
                    return False
        return True

    def get_balanced_set(self):
        """
        Returns the fully collected and balanced list of questions if complete,
        otherwise returns None.
        """
        if not self.is_full():
            return None  # Not enough data yet

        all_qs = []
        for choice in ["A", "B", "C"]:
            for difficulty in ["easy", "medium", "hard"]:
                all_qs.extend(self.data[choice][difficulty])
        return all_qs


# pylint: disable=too-many-instance-attributes
class SaveBenchmark:
    """Saves the benchmark (blank or completed) as a .npz"""

    def __init__(self, path, exam_name, **kwargs):
        self.path = path  # path including /PATH/benchmark_results/blank/
        # (for blank benchmark) or including /PATH/benchmark_results/completed/
        # (for benchmarked model results)
        self.exam_name = exam_name
        self.checkpoint_freq = kwargs.get("checkpoint_freq", "unset")
        self.restart_idx = kwargs.get("restart_idx", "unset")
        self.model = kwargs.get("model", "no_model")
        self.exam_idx = kwargs.get("exam_idx") or "unset"  # ,'unset')
        self.agent = kwargs.get("agent", False)
        # self.responses = kwargs.get('model', 'no_model')
        self.save_npz_path = self.path
        # create_missing_directory(self.save_npz_path)
        self.attributes_to_save = []

        # Determine filename
        self.npz_filename = get_npz_filename_no_model(
            self.save_npz_path, self.exam_name, self.exam_idx
        )

        if "_" in self.exam_name:
            self.ci_method = (self.exam_name).split("_")[1]
            self.exam_name_wo_ci_method = (self.exam_name).split("_")[0]
        else:
            self.exam_name_wo_ci_method = self.exam_name

        # Necessary so that these attributes can be set on the instance
        # of the corresponding generated benchmarks:
        self.question = None
        self.solution = None
        self.difficulty = None
        self.p_diff = None
        self.p_diff_ci_lower = None
        self.p_diff_ci_upper = None
        self.n_samples = None
        self.table = None
        self.name = None
        self.response = None
        self.grade = None
        self.unbiased_solution = None
        self.biased_solution = None

    @classmethod
    def from_simple_inequality(cls, source, path, exam_name, exam_idx):
        """Constructs a new instance from the class SaveBenchmark and then
        set the source (SimpleInequality) attributes on the instance"""
        instance = cls(path, exam_name)
        instance.exam_idx = exam_idx
        instance.save_npz_path = path
        # Build filename
        if exam_idx != "unset":
            filename = f"{exam_name}_{exam_idx}.npz"
        # elif exam_idx == None:
        #    filename = f"{exam_name}.npz"
        else:
            filename = f"{exam_name}.npz"
        instance.npz_filename = os.path.join(instance.save_npz_path, filename)
        # Shuffle the questions:
        # Get number of samples (assumes all arrays are the same length)
        n = len(source.question)
        # Generate a random permutation of indices
        perm = np.random.permutation(n)
        # Apply the permutation to all arrays
        instance.question = np.array(source.question)[perm]
        instance.solution = np.array(source.solution)[perm]
        instance.difficulty = np.array(source.difficulty)[perm]
        instance.mean_diff = np.array(source.mean_diff)[perm]
        instance.ci_lower = np.array(source.ci_lower)[perm]
        instance.ci_upper = np.array(source.ci_upper)[perm]
        instance.n_samples = np.array(source.n_samples)[perm]
        instance.name = np.array(source.name)[perm]
        return instance

    @classmethod
    def from_complex_inequality(cls, source, path, exam_name, exam_idx):
        """Constructs a new instance from the class SaveBenchmark and then
        set the source (ComplexInequality) attributes on the instance"""
        instance = cls(path, exam_name)
        instance.exam_idx = exam_idx
        instance.save_npz_path = path
        # Build filename
        if exam_idx != "unset":
            filename = f"{exam_name}_{exam_idx}.npz"
        # elif exam_idx == None:
        #    filename = f"{exam_name}.npz"
        else:
            filename = f"{exam_name}.npz"
        instance.npz_filename = os.path.join(instance.save_npz_path, filename)
        # Shuffle the questions:
        # Get number of samples (assumes all arrays are the same length)
        n = len(source.question)
        # Generate a random permutation of indices
        perm = np.random.permutation(n)
        # Apply the permutation to all arrays
        instance.question = np.array(source.question)[perm]
        instance.solution = np.array(source.solution)[perm]
        instance.difficulty = np.array(source.difficulty)[perm]
        instance.mean_diff = np.array(source.mean_diff)[perm]
        instance.ci_lower = np.array(source.ci_lower)[perm]
        instance.ci_upper = np.array(source.ci_upper)[perm]
        instance.n_samples = np.array(source.n_samples)[perm]
        instance.name = np.array(source.name)[perm]
        return instance

    @classmethod
    def from_standard_deviation(cls, source, path, exam_name, exam_idx):
        """Constructs a new instance from the class SaveBenchmark and then
        set the source (StandardDeviation) attributes on the instance"""
        instance = cls(path, exam_name)
        instance.exam_idx = exam_idx
        instance.save_npz_path = path
        # Build filename
        if exam_idx != "unset":
            filename = f"{exam_name}_{exam_idx}.npz"
        else:
            filename = f"{exam_name}.npz"
        instance.npz_filename = os.path.join(instance.save_npz_path, filename)
        # Shuffle the questions:
        # Get number of samples (assumes all arrays are the same length)
        n = len(source.question)
        # Generate a random permutation of indices
        perm = np.random.permutation(n)
        # Apply the permutation to all arrays
        instance.question = np.array(source.question)[perm]
        instance.biased_solution = np.array(source.biased_solution)[perm]
        instance.unbiased_solution = np.array(source.unbiased_solution)[perm]
        instance.name = np.array(source.name)[perm]
        return instance

    @classmethod
    def from_mediated_causality(cls, source, path, exam_name, exam_idx):
        """Constructs a new instance from the class SaveBenchmark and then
        set the source (MediatedCausality) attributes on the instance"""
        instance = cls(path, exam_name)
        instance.exam_idx = exam_idx
        instance.save_npz_path = path
        # Build filename
        if exam_idx != "unset":
            filename = f"{exam_name}_{exam_idx}.npz"
        else:
            filename = f"{exam_name}.npz"
        instance.npz_filename = os.path.join(instance.save_npz_path, filename)
        # Shuffle the questions:
        # Get number of samples (assumes all arrays are the same length)
        n = len(source.question)
        # Generate a random permutation of indices
        perm = np.random.permutation(n)
        # Apply the permutation to all arrays
        instance.question = np.array(source.question)[perm]
        instance.solution = np.array(source.solution)[perm]
        instance.difficulty = np.array(source.difficulty)[perm]
        instance.p_diff = np.array(source.p_diff)[perm]
        instance.p_diff_ci_lower = np.array(source.p_diff_ci_lower)[perm]
        instance.p_diff_ci_upper = np.array(source.p_diff_ci_upper)[perm]
        instance.n_samples = np.array(source.n_samples)[perm]
        instance.table = np.array(source.table)[perm]
        instance.name = np.array(source.name)[perm]
        return instance

    # @classmethod
    def save_attributes(self):
        """Save the data as a dict within an npz file"""
        if self.exam_name_wo_ci_method in (
            "MediatedCausalitySmoking",
            "MediatedCausality",
            "MediatedCausalityWithMethod",
        ):
            self.attributes_to_save = [
                "question",
                "solution",
                "difficulty",
                "p_diff",
                "p_diff_ci_lower",
                "p_diff_ci_upper",
                "n_samples",
                "table",
                "name",
            ]
            data = {key: getattr(self, key) for key in self.attributes_to_save}
            np.savez(self.npz_filename, **data)
        elif self.exam_name_wo_ci_method in (
            "SimpleInequality",
            "SimpleInequalityAgent",
            "SimpleInequalityWithMethod",
            "ComplexInequality",
            "ComplexInequalityWithMethod",
        ):
            self.attributes_to_save = [
                "question",
                "solution",
                "difficulty",
                "mean_diff",
                "ci_lower",
                "ci_upper",
                "n_samples",
                "name",
            ]
        elif self.exam_name_wo_ci_method in ("StandardDeviation"):
            self.attributes_to_save = [
                "question",
                "biased_solution",
                "unbiased_solution",
                "name",
            ]
        else:
            raise ValueError(f"Unsupported benchmark: {self.exam_name_wo_ci_method}")
        #    data = {key: getattr(self, key) for key in self.attributes_to_save}
        #    for key in self.attributes_to_save:
        #        value = getattr(self, key, None)
        #        if value is None:
        #            raise ValueError(f"Attribute '{key}' is None and cannot be saved.")
        #    np.savez(self.npz_filename, **data)
        data = {}
        for key in self.attributes_to_save:
            value = getattr(self, key, None)
            if value is None:
                print(f" Warning: '{key}' is None and will not be saved.")
            else:
                data[key] = value
        np.savez(self.npz_filename, **data)
