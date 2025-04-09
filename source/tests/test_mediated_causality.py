# tests/test_mediated_causality.py

import numpy as np
from source.benchmarks.mediatedCausality import mediatedCausality

def test_causality_from_table_tdist():
    # Verifies the front-door criterion calculation
    exam = mediatedCausality(
        './',
        'mediatedCausalitySmoking_tdist',
        generate_flag=False
    )
    # This table is equivalent to the table on p.84 of Pearl "Causality":
    pearl_table = np.array([[0,0,0,427.50],
                            [0,0,1, 23.75],
                            [0,1,0, 47.50],
                            [0,1,1,  1.25],
                            [1,0,0,  2.50],
                            [1,0,1, 71.25],
                            [1,1,0, 22.50],
                            [1,1,1,403.75]])
    p_diff,p_diff_ci_lower,p_diff_ci_upper = exam.causality_from_table(pearl_table, 'tdist')
    assert p_diff == (0.4525 - 0.4975)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.08872766149704325,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.00127233850295672,3)

def test_causality_from_table_bootstrap():
    # Verifies the front-door criterion calculation
    exam = mediatedCausality(
        './',
        'mediatedCausalitySmoking_bootstrap',
        generate_flag=False,
        n_bootstrap=500
    )
    # This table is equivalent to the table on p.84 of Pearl "Causality":
    pearl_table = np.array([[0,0,0,42750],
                            [0,0,1, 2375],
                            [0,1,0, 4750],
                            [0,1,1,  125],
                            [1,0,0,  250],
                            [1,0,1, 7125],
                            [1,1,0, 2250],
                            [1,1,1,40375]])
    p_diff,p_diff_ci_lower,p_diff_ci_upper = exam.causality_from_table(pearl_table, 'bootstrap')
    assert np.round(p_diff,3) == np.round(0.4525 - 0.4975,3)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.05151729437221199,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.03808042079620048,3)

def test_output_dims():
    # This test verifies that the causal calculation is correct
    n_problems = 9
    exam_name = 'mediatedCausalitySmoking_tdist'
    plot_path = './figures/'
    exam = mediatedCausality(plot_path, exam_name, n_problems=n_problems)
    metadata = exam.get_metadata()
    solutions = exam.get_solutions()
    questions = exam.get_questions()
    print(np.count_nonzero(solutions == 'A'))
    assert np.shape(solutions)[0] == n_problems
    assert np.shape(questions)[0] == n_problems
    assert np.shape(metadata["P_diff"])[0] == n_problems
    assert np.shape(metadata["P_diff_CIu"])[0] == n_problems
    assert np.shape(metadata["P_diff_CIl"])[0] == n_problems
    assert np.shape(metadata["n_samples"])[0] == n_problems
    assert np.shape(metadata["difficulty"])[0] == n_problems
    assert np.count_nonzero(solutions == 'A') == int(n_problems/3)
    assert np.count_nonzero(solutions == 'B') == int(n_problems/3)
    assert np.count_nonzero(solutions == 'C') == int(n_problems/3)
    assert np.count_nonzero(metadata["difficulty"] == 'easy') == int(n_problems/3)
    assert np.count_nonzero(metadata["difficulty"] == 'intermediate') == int(n_problems/3)
    assert np.count_nonzero(metadata["difficulty"] == 'difficult') == int(n_problems/3)
