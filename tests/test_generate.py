""" Tests for Generate """
import os
import pytest
import numpy as np
from tether.benchmarks.mediated_causality import causality_from_table
from tether.core.generator import generate_benchmarks

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.

@pytest.fixture
def user_specific_path():
    path = os.environ.get("PATH_TO_BENCHMARKS")
    if not path or not os.path.exists(path):
        pytest.fail("PATH_TO_BENCHMARKS not set in environment.")
    return path

# could test all of the benchmarks:
def test_generator_shuffle(user_specific_path):
    """ Verifies that the shuffling of problems within SaveBenchmark
    inside Generator is working properly (arrays shuffled together) """
    exam_name = 'MediatedCausalitySmoking_tdist'
    exam_idx=99999 # set to prevent overwriting other exams
    os.makedirs(os.path.join(user_specific_path, "blank"), exist_ok=True)
    generate_benchmarks(user_specific_path, exam_name, n_problems=9, exam_idx=exam_idx)
    filename = os.path.join(user_specific_path, "blank", f"{exam_name}_{exam_idx}.npz")
    data = np.load(filename, allow_pickle=True)
    n = len(data['question'])
    for i in range(0,n):
        table = data['table'][i,:,:]
        result = causality_from_table(table, 'tdist')
        p_diff_verify = result[:1]
        assert np.allclose(data['p_diff'][i], p_diff_verify, atol=1e-4)
