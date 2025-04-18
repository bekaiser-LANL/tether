""" Tests for Generate """
import os
import numpy as np
from source.benchmarks.mediated_causality import causality_from_table
from source.generator import Generator

def test_generator_shuffle():
    """ Verifies that the shuffling of problems within SaveBenchmark
    inside Generator is working properly (arrays shuffled together)"""
    exam_name = 'MediatedCausalitySmoking_tdist'
    path = '/Users/l281800/Desktop/benchmarks/saved/'
    exam_idx=99999 # set to prevent overwriting other exams
    Generator(path, exam_name, n_problems=9, exam_idx=exam_idx)
    filename = os.path.join(path, f"{exam_name}_{exam_idx}.npz")
    #Generator(path, exam_name, n_problems=9)
    #filename = os.path.join(path, f"{exam_name}.npz")
    data = np.load(filename, allow_pickle=True)
    n = len(data['question'])
    for i in range(0,n):
        table = data['table'][i,:,:]
        result = causality_from_table(table, 'tdist')
        p_diff_verify = result[:1]
        assert np.allclose(data['p_diff'][i], p_diff_verify, atol=1e-4)
