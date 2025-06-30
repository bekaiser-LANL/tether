""" Tests for StandardDeviation """
import numpy as np
from tether.benchmarks.standard_deviation import StandardDeviation

def test_initialization_defaults():
    """ Test that default parameters are set correctly """
    sd = StandardDeviation(exam_name='TestExam')
    assert isinstance(sd.name, np.ndarray)
    assert len(sd.name) == sd.n_problems
    assert np.all(sd.name == 'TestExam') 
    assert sd.generate_flag is True
    assert sd.verbose is False
    assert sd.n_problems == 100
    assert sd.number_range == [-100., 100.]
    assert sd.n_numbers == 10

def test_generate_question():
    """ Test that generate_question returns proper 
    random numbers and formatted string """
    sd = StandardDeviation(exam_name='TestExam', generate_flag=False)
    random_numbers, q_str = sd.generate_question()

    assert isinstance(random_numbers, np.ndarray)
    assert len(random_numbers) == sd.n_numbers
    assert isinstance(q_str, str)
    assert q_str.startswith("What is the standard deviation of")

def test_make_problems_length():
    """ Test that make_problems generates the 
    correct number of problems """
    n_problems = 5
    sd = StandardDeviation(exam_name='TestExam', n_problems=n_problems)

    assert len(sd.question) == n_problems
    assert len(sd.biased_solution) == n_problems
    assert len(sd.unbiased_solution) == n_problems
    assert len(sd.example_idx) == n_problems

def test_standard_deviation_calculation_accuracy():
    """ Test that biased and unbiased standard 
    deviation are calculated correctly """
    sd = StandardDeviation(exam_name='TestExam', generate_flag=False)
    test_numbers = np.array([1, 2, 3, 4, 5])
    sd.n_numbers = len(test_numbers)

    # Patch generate_question to return our test numbers
    def fake_generate_question():
        q_str = "What is the standard deviation of 1 2 3 4 5?"
        return test_numbers, q_str

    sd.generate_question = fake_generate_question
    sd.n_problems = 1
    sd.make_problems()

    biased_expected = '{:.{}f}'.format(np.std(test_numbers), 10)
    unbiased_expected = '{:.{}f}'.format(np.std(test_numbers, ddof=1), 10)

    assert sd.biased_solution[0] == biased_expected
    assert sd.unbiased_solution[0] == unbiased_expected
