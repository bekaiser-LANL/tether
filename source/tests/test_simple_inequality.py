import pytest
import numpy as np
from source.benchmarks.simple_inequality import SimpleInequality

@pytest.fixture
def simple_inequality_instance():
    return SimpleInequality(
        exam_name="TestSimple",
        n_numbers=10,
        n_problems=9,
        generate_flag=False  # Prevent auto generation
    )

def test_generate_vector_stats(simple_inequality_instance):
    vec = simple_inequality_instance.generate_vector(target_mean=0.5, target_std=0.2, length=10)
    assert isinstance(vec, np.ndarray)
    assert len(vec) == 10
    assert np.isclose(np.mean(vec), 0.5, atol=0.2)
    assert np.isclose(np.std(vec), 0.2, atol=0.1)

def test_generate_vector_pair_diff_range(simple_inequality_instance):
    vec1, vec2, std1, std2 = simple_inequality_instance.generate_vector_pair((0.4, 0.6))
    diff = abs(np.mean(vec1) - np.mean(vec2))
    assert 0.4 <= diff <= 0.6
    assert isinstance(vec1, np.ndarray)
    assert isinstance(vec2, np.ndarray)

def test_find_mean_difference(simple_inequality_instance):
    vec1 = np.array([1, 2, 3])
    vec2 = np.array([4, 5, 6])
    mean1, mean2, diff = simple_inequality_instance.find_mean_difference(vec1, vec2)
    assert np.isclose(mean1, 2.0)
    assert np.isclose(mean2, 5.0)
    assert np.isclose(diff, 3.0)

def test_assign_difficulty(simple_inequality_instance):
    hard_vec = np.array([0.1] * 10)
    easy_vec = np.array([2.0] * 10)
    assert simple_inequality_instance.assign_difficulty(hard_vec, hard_vec) == 'hard'
    assert simple_inequality_instance.assign_difficulty(hard_vec, easy_vec) == 'easy'

def test_calculate_ci_structure(simple_inequality_instance):
    vec1 = np.random.normal(0.5, 0.1, 10)
    vec2 = np.random.normal(-0.5, 0.1, 10)
    _, _, std1, std2 = simple_inequality_instance.generate_vector_pair((0.5, 0.6))
    simple_inequality_instance.n_numbers = 10  # ensure correct n
    ci_lower, ci_upper, diff = simple_inequality_instance.calculate_ci(vec1, vec2, (0.5, 0.6))
    assert ci_lower < ci_upper
    assert isinstance(diff, float)

def test_record_solutions_labels(simple_inequality_instance):
    assert simple_inequality_instance.record_solutions(0.1, 0.5)[1] == 'A'
    assert simple_inequality_instance.record_solutions(-0.5, -0.1)[1] == 'B'
    assert simple_inequality_instance.record_solutions(-0.2, 0.2)[1] == 'C'

