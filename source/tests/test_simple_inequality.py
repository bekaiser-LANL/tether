import numpy as np
import pytest
from source.benchmarks.simple_inequality import SimpleInequality

@pytest.fixture
def si_instance():
    return SimpleInequality(exam_name="simpleInequalityTest", n_problems=9, generate_flag=False)

def test_generate_vector_shape_and_stats(si_instance):
    vec = si_instance.generate_vector(target_mean=0.5, target_std=0.2, length=20)
    assert isinstance(vec, np.ndarray)
    assert vec.shape[0] == si_instance.n_numbers
    assert abs(np.mean(vec) - 0.5) < 0.1
    assert abs(np.std(vec) - 0.2) < 0.1

def test_generate_vector_pair(si_instance):
    v1, v2 = si_instance.generate_vector_pair((0.5, 1.0))
    assert isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray)
    assert len(v1) == len(v2) == si_instance.n_numbers
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)
    assert abs(mean1 - mean2) >= 0.5
    assert abs(mean1 - mean2) <= 1.0

def test_generate_dataset_range_cycling(si_instance):
    ranges = si_instance.mean_diff_ranges
    for i in range(len(ranges) * 2):  # Test cycling
        v1, v2 = si_instance.generate_dataset()
        mean1 = np.mean(v1)
        mean2 = np.mean(v2)
        diff = abs(mean1 - mean2)
        current_range = ranges[i % len(ranges)]
        assert current_range[0] <= diff <= current_range[1]

def test_find_mean_difference(si_instance):
    v1 = np.array([1, 2, 3])
    v2 = np.array([3, 4, 5])
    mean1, mean2, diff = si_instance.find_mean_difference(v1, v2)
    assert mean1 == pytest.approx(2.0)
    assert mean2 == pytest.approx(4.0)
    assert diff == pytest.approx(2.0)

def test_record_solutions(si_instance):
    v1 = np.array([5, 5, 5])
    v2 = np.array([1, 1, 1])
    _, answer = si_instance.record_solutions(v1, v2)
    assert answer == 'A'

    v1 = np.array([1, 1, 1])
    v2 = np.array([5, 5, 5])
    _, answer = si_instance.record_solutions(v1, v2)
    assert answer == 'B'

    v1 = v2 = np.array([2, 2, 2])
    _, answer = si_instance.record_solutions(v1, v2)
    assert answer == 'C'

def test_assign_difficulty(si_instance):
    thresholds = si_instance.difficulty_thresholds
    # Below first threshold
    v1 = np.array([0.1, 0.1, 0.1])
    v2 = np.array([0.1 + thresholds[0] / 2] * 3)
    difficulty = si_instance.assign_difficulty(v1, v2)
    assert difficulty == 'hard'

    # Between thresholds
    v2 = np.array([0.1 + (thresholds[0] + 0.01)] * 3)
    difficulty = si_instance.assign_difficulty(v1, v2)
    assert difficulty == 'medium'

    # Above second threshold
    v2 = np.array([0.1 + thresholds[1] + 0.1] * 3)
    difficulty = si_instance.assign_difficulty(v1, v2)
    assert difficulty == 'easy'
