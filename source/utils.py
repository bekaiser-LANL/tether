""" General purpose benchmark functions & classes """
import random
import os
import numpy as np

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
