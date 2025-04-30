import numpy as np
import pytest
from source.benchmarks.complex_inequality import ComplexInequality

# Fixture to create a basic instance
@pytest.fixture
def ci_instance():
    return ComplexInequality(
        exam_name="ComplexInequality_tdist",
        n_numbers=10,
        n_problems=9,
        generate_flag=False  # prevent auto problem generation during testing
    )

def test_find_mean_difference(ci_instance):
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    mean1, mean2, diff = ci_instance.find_mean_difference(vec1, vec2)
    assert np.isclose(mean1, 2.0)
    assert np.isclose(mean2, 5.0)
    assert np.isclose(diff, -3.0)

def test_assign_difficulty(ci_instance):
    # Based on default thresholds: [0.66, 1.33]
    hard_vec = np.array([0.1] * 10)
    easy_vec = np.array([2.0] * 10)
    assert ci_instance.assign_difficulty(hard_vec, hard_vec) == 'hard'
    assert ci_instance.assign_difficulty(hard_vec, easy_vec) == 'easy'

def test_calculate_ci(ci_instance):
    vec1 = np.ones(10)
    vec2 = np.zeros(10)
    std1, std2 = 0.1, 0.1
    ci_lower, ci_upper, diff = ci_instance.calculate_ci(vec1, vec2, std1, std2)
    assert diff > 0
    assert ci_lower < diff < ci_upper

def test_bootstrap_ci(ci_instance):
    vec1 = np.ones(10)
    vec2 = np.zeros(10)
    ci_lower, ci_upper, diff = ci_instance.bootstrap_ci(vec1, vec2)
    assert ci_lower <= diff <= ci_upper
    assert ci_upper >= ci_lower

def test_record_solutions():
    ci = ComplexInequality("ComplexInequality_tdist", n_problems=9, generate_flag=False)
    assert ci.record_solutions(1, 2)[1] == 'A'
    assert ci.record_solutions(-2, -1)[1] == 'B'
    assert ci.record_solutions(-1, 1)[1] == 'C'

def test_get_prompts_structure(ci_instance):
    vec1, vec2, question, chosen_range, std1, std2 = ci_instance.get_prompts()
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)
    assert isinstance(question, str)
    assert isinstance(chosen_range, tuple)
    assert isinstance(std1, float)
    assert isinstance(std2, float)

def test_generate_vector_pair(ci_instance):
    vec1, vec2, std1, std2 = ci_instance.generate_vector_pair((0.5, 1.0))
    assert len(vec1) == ci_instance.n_numbers
    assert len(vec2) == ci_instance.n_numbers
    assert isinstance(std1, float)
    assert isinstance(std2, float)

def test_prob_greater_than_from_pdf(ci_instance):
    x = np.linspace(-10, 10, 1000)
    pdf = ci_instance.gaussian(x)
    prob = ci_instance.prob_greater_than_from_pdf(x, pdf, 0)
    assert np.isclose(prob, 0.5, atol=0.05)
