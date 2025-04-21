""" General purpose benchmark functions & classes """
import os
import numpy as np

def strip_after_second_underscore(s):
    """ Get the exam_name if it has exam_idx on the end.
    For names like MediatedCausality_tdist_0 it will
    extract the exam_name MediatedCausality_tdist """
    parts = s.split("_")
    if len(parts) >= 2:
        return "_".join(parts[:2])
    return s

def get_after_second_underscore(s):
    """ Get the exam_idx if the exam_name has one.
    For names like MediatedCausality_tdist_0 it will
    extract the exam_name MediatedCausality_tdist """
    parts = s.split("_")
    if len(parts) > 2:
        return "_".join(parts[2:])
    return ""

def get_npz_filename_no_model(save_npz_path, exam_name, exam_idx):
    """ Determine filename """
    if exam_idx != 'unset':
        filename = f"{exam_name}_{exam_idx}.npz"  
    else:
        filename = f"{exam_name}.npz"
    npz_filename = os.path.join(
        save_npz_path, 
        filename
    )
    return npz_filename

def get_npz_filename(save_npz_path, exam_name, exam_idx, model):
    """ Determine filename """
    if exam_idx != 'unset':
        filename = f"{exam_name}_{model}_{exam_idx}.npz"  
    else:
        filename = f"{exam_name}_{model}.npz"
    npz_filename = os.path.join(
        save_npz_path, 
        filename
    )
    return npz_filename

def load_saved_benchmark(benchmark_path, exam_name, exam_idx):
    """ read saved .npz file containing dict (benchmark) """
    if exam_idx != 'unset':
        filename = os.path.join(benchmark_path, f"{exam_name}_{exam_idx}.npz")
    else:
        filename = os.path.join(benchmark_path, f"{exam_name}.npz") 
    data = np.load(filename, allow_pickle=True)
    return data

def is_integer(value):
    return isinstance(value, int)

def get_95_CI_tdist(P,N):
    # t distribution to estimate standard error
    se = standard_error_for_proportion(P,N) 
    return P+1.96*se,P-1.96*se

def standard_error_for_proportion(P,N):
    # Brayer, Edward F. "Calculating the standard error of a proportion." 
    # Journal of the Royal Statistical Society Series C: Applied Statistics 6.1 (1957): 67-68.
    return np.sqrt((P*(1.-P))/N) 

def check_probability(P):
    if P > 1.:
        print('\n ERROR: Probability > 1')
    elif P < 0.:
        print('\n ERROR: Probability < 0')
    return

def enforce_probability_bounds(var):
    if var > 1.:
        var = 1.
    elif var < 0.:
        var = 0.
    return var

def create_missing_directory(directory_path):
    """ Checks if a directory exists and makes it if not """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def is_divisible_by_9(number):
    """ Checks if divisible by 9 """
    return number % 9 == 0

def is_divisible_by_3(number):
    """ Checks if divisible by 3 """
    return number % 3 == 0

class ReadSavedBenchmarkNpz():
    def __init__(self, read_path  ):
        self.read_path = read_path
  
        data = np.load(read_path,allow_pickle=True)
        self.exam_name = data['exam_name']
        self.questions = data['questions']
        self.solutions = data['solutions']
        self.model_str = data['model_str']
        self.exam_str = data['exam_str']
        self.n_problems = data['n_problems']

        if self.exam_name == 'significantFigures' or self.exam_name == 'standardDeviation':
            self.metadata = {
                "Name": self.exam_name,
                "n_problems": self.n_problems
            }
        elif self.exam_name == 'mediatedCausalitySmoking' or self.exam_name == 'mediatedCausalitySmokingWithMethod':
            self.metadata = {
                "Name": self.exam_name,
                "dP": data['dP'],
                "P_Y1doX1": data['P_Y1doX1'],
                "P_Y1doX0": data['P_Y1doX0'],
                "P_Y1doX1_CI": data['P_Y1doX1_CI'],
                "P_Y1doX0_CI": data['P_Y1doX0_CI'],
                "A_count": data['C_count'],
                "B_count": data['C_count'],
                "C_count": data['C_count'],
                "n_problems": self.n_problems
            }

    def get_questions(self): # all tests need this
        return self.questions

    def get_solutions(self): # all tests need this
        return self.solutions

    def get_metadata(self): # all tests need this
        return self.metadata
    
class QuestionBank:
    def __init__(self, target_per_bin=1):
        """
        Initialize a tracker for question generation.

        Parameters:
            target_per_bin (int): How many questions 
            you want per (choice, difficulty) bin.
        """
        self.target_per_bin = target_per_bin

        self.data = {
            choice: {difficulty: [] for difficulty in ['easy', 'medium', 'hard']}
            for choice in ['A', 'B', 'C']
        }

    def add_question(self, question_text, correct_choice, difficulty, metadata=None):
        """
        Adds a question to the appropriate bin.
        """
        if correct_choice not in self.data:
            raise ValueError("Invalid correct choice. Must be 'A', 'B', or 'C'")
        if difficulty not in self.data[correct_choice]:
            raise ValueError("Invalid difficulty. Must be 'easy', 'medium', or 'hard'")

        bin_list = self.data[correct_choice][difficulty]
        if len(bin_list) >= self.target_per_bin:
            # Skip adding to overfilled bin
            return False  # Optionally indicate it was rejected

        entry = {
            'question': question_text,
            'solution': correct_choice,
            'difficulty': difficulty,
            'metadata': metadata or {}
        }

        bin_list.append(entry)
        return True

    def count(self):
        """
        Returns a nested count of how many questions are in each bin.
        """
        return {
            choice: {
                difficulty: len(self.data[choice][difficulty])
                for difficulty in self.data[choice]
            } for choice in self.data
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
        for choice in ['A', 'B', 'C']:
            for difficulty in ['easy', 'medium', 'hard']:
                all_qs.extend(self.data[choice][difficulty])
        return all_qs

class SaveBenchmark():
    """ Saves the benchmark (blank or completed) as a .npz"""

    def __init__(self, path, exam_name, **kwargs):
        self.path = path # path including /PATH/benchmarks/saved/ 
        # (for blank benchmark) or including /PATH/benchmarks/results/
        # (for benchmarked model results)
        self.exam_name = exam_name

        self.checkpoint_freq = kwargs.get('checkpoint_freq','unset')
        self.restart_idx = kwargs.get('restart_idx','unset')
        self.model = kwargs.get('model', 'no_model')
        self.exam_idx = kwargs.get('exam_idx','unset')
        #self.responses = kwargs.get('model', 'no_model')

        # Determine save path based on model type
        #if self.model == 'no_model':
        #    subfolder = 'saved'
        #else: 
        #    subfolder = os.path.join('results', self.model)
        #self.save_npz_path = os.path.join(self.path, subfolder)
        self.save_npz_path = self.path       
        #create_missing_directory(self.save_npz_path)

        # Determine filename 
        self.npz_filename = get_npz_filename_no_model(
            self.save_npz_path,
            self.exam_name,
            self.exam_idx
        )
        # if self.exam_idx != 'unset':
        #     filename = f"{self.exam_name}_{self.exam_idx}.npz"  
        # else:
        #     filename = f"{self.exam_name}.npz"
        # self.npz_filename = os.path.join(
        #     self.save_npz_path, 
        #     filename
        # )

        if '_' in self.exam_name:
            self.ci_method = (self.exam_name).split('_')[1]
            self.exam_name_wo_ci_method = (self.exam_name).split('_')[0]
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

    @classmethod
    def from_simple_inequality(cls, source, path, exam_name):
        """ Constructs a new instance from the class SaveBenchmark and then
        set the source (SimpleInequality) attributes on the instance """
        # TO DO        
        return cls(name=source.name, age=source.age) # <-

    @classmethod
    def from_complex_inequality(cls, source, path, exam_name):
        """ Constructs a new instance from the class SaveBenchmark and then
        set the source (ComplexInequality) attributes on the instance """
        # TO DO        
        return cls(name=source.name, age=source.age) # <-

    @classmethod
    def from_significant_figures(cls, source, path, exam_name):
        """ Constructs a new instance from the class SaveBenchmark and then
        set the source (SignificantFigures) attributes on the instance """
        # TO DO        
        return cls(name=source.name, age=source.age) # <-

    @classmethod
    def from_standard_deviation(cls, source, path, exam_name):
        """ Constructs a new instance from the class SaveBenchmark and then
        set the source (StandardDeviation) attributes on the instance """
        # TO DO
        return cls(title=source.title)

    @classmethod
    def from_mediated_causality(cls, source, path, exam_name, exam_idx):
        """ Constructs a new instance from the class SaveBenchmark and then
        set the source (MediatedCausality) attributes on the instance """
        instance = cls(path, exam_name)
        instance.exam_idx = exam_idx
        instance.save_npz_path = path
        # Build filename
        if exam_idx != 'unset':
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
    
    @classmethod
    def save_attributes(self):
        """ Save the data as a dict within an npz file"""
        if self.exam_name_wo_ci_method in (
            'MediatedCausalitySmoking', 
            'MediatedCausality'
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
                "name"
            ]
            data = {key: getattr(self, key) for key in self.attributes_to_save}
            np.savez(self.npz_filename, **data)
        else:
            raise ValueError(
                f"Unsupported benchmark: {self.exam_name_wo_ci_method}"
            )
