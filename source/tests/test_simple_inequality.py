import numpy as np
import pytest
from scipy.integrate import trapezoid
from source.benchmarks.simple_inequality import SimpleInequality  # Adjust path as needed

@pytest.fixture
def si_instance():
    return SimpleInequality(plot_path=".", exam_name="SimpleInequality_tdist", n_numbers=100, generate_flag=False)

def test_generate_vector_shape(si_instance):
    vec = si_instance.generate_vector(target_mean=0, target_std=1, length=100)
    assert len(vec) == 100

def test_generate_vector_pair_diff_in_range(si_instance):
    vec1, vec2, std1, std2 = si_instance.generate_vector_pair((0.5, 1.0))
    diff = abs(np.mean(vec1) - np.mean(vec2))
    assert 0 <= diff <= 3.0

def test_generate_dataset_output(si_instance):
    _, vec1, vec2, std1, std2 = si_instance.generate_dataset()
    assert isinstance(std1, float)
    assert isinstance(std2, float)
    assert len(vec1) == si_instance.n_numbers

def test_calculate_ci_order(si_instance):
    vec1 = np.random.normal(0, 1, si_instance.n_numbers)
    vec2 = np.random.normal(1, 1, si_instance.n_numbers)
    std1, std2 = np.std(vec1), np.std(vec2)
    ci_lower, ci_upper, diff = si_instance.calculate_ci(vec1, vec2, std1, std2)
    assert ci_lower < ci_upper
    assert isinstance(diff, float)

def test_bootstrap_ci_returns_bounds(si_instance):
    vec1 = np.random.normal(0, 1, si_instance.n_numbers)
    vec2 = np.random.normal(1, 1, si_instance.n_numbers)
    ci_lower, ci_upper, diff = si_instance.bootstrap_ci(vec1, vec2)
    assert ci_lower < ci_upper
    assert isinstance(diff, float)

def test_record_solutions_logic(si_instance):
    assert si_instance.record_solutions(0.5, 1.5)[1] == 'A'
    assert si_instance.record_solutions(-2.0, -1.0)[1] == 'B'
    assert si_instance.record_solutions(-1.0, 1.0)[1] == 'C'

def test_find_mean_difference(si_instance):
    v1 = np.array([1, 2, 3])
    v2 = np.array([0, 0, 0])
    mean1, mean2, diff = si_instance.find_mean_difference(v1, v2)
    assert np.isclose(mean1, 2.0)
    assert np.isclose(mean2, 0.0)
    assert np.isclose(diff, 2.0)

def test_assign_difficulty(si_instance):
    assert si_instance.assign_difficulty(0.3) == 'hard'
    assert si_instance.assign_difficulty(1.0) == 'medium'
    assert si_instance.assign_difficulty(2.0) == 'easy'
    assert si_instance.assign_difficulty(np.nan) == 'N/A'

