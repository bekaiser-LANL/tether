# tests/test_mediatedCausality.py

import pytest
import numpy as np
from source.benchmarks.mediatedCausality import mediatedCausality

def test_causality_from_table():
    # This test verifies that the causal calculation is correct
    exam = mediatedCausality('./','mediatedCausalitySmoking',generate_flag=False)
    # This table is equivalent to the table on p.84 of Pearl "Causality":
    pearl_table = np.array([[0,0,0,427.50],
                            [0,0,1, 23.75],
                            [0,1,0, 47.50],
                            [0,1,1,  1.25],
                            [1,0,0,  2.50],
                            [1,0,1, 71.25],
                            [1,1,0, 22.50],
                            [1,1,1,403.75]])
    P_Y1doX1,a_,b_,P_Y1doX0,c_,d_,e_ = exam.causality_from_table(pearl_table)
    assert P_Y1doX1 == 0.4525
    assert P_Y1doX0 == 0.4975

def test_output_dims():
    # This test verifies that the causal calculation is correct
    n_problems_ = 18
    exam_name = 'mediatedCausalitySmoking'
    plot_path = './figures/'
    exam = mediatedCausality(plot_path, exam_name, n_problems=n_problems_)
    metadata = exam.get_metadata()
    solutions = exam.get_solutions()
    questions = exam.get_questions()
    print(np.count_nonzero(solutions == 'A'))
    assert np.shape(solutions)[0] == n_problems_
    assert np.shape(questions)[0] == n_problems_
    assert np.shape(metadata["P_Y1doX1"])[0] == n_problems_
    assert np.shape(metadata["P_Y1doX0"])[0] == n_problems_
    assert np.shape(metadata["P_Y1doX1_CI"])[0] == n_problems_
    assert np.shape(metadata["P_Y1doX0_CI"])[0] == n_problems_   
    assert np.shape(metadata["n_samples"])[0] == n_problems_   
    assert np.shape(metadata["difficulty"])[0] == n_problems_  
    assert np.count_nonzero(solutions == 'A') == int(n_problems_/3)
    assert np.count_nonzero(solutions == 'B') == int(n_problems_/3)
    assert np.count_nonzero(solutions == 'C') == int(n_problems_/3)                  
    assert np.count_nonzero(metadata["difficulty"] == 'easy') == int(n_problems_/3)
    assert np.count_nonzero(metadata["difficulty"] == 'intermediate') == int(n_problems_/3)
    assert np.count_nonzero(metadata["difficulty"] == 'difficult') == int(n_problems_/3)   


# add a test that determines if the text of the problem matches the table