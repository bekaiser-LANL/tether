# tests/test_mediated_causality.py
import re
import pytest
import numpy as np
from source.benchmarks.mediated_causality import MediatedCausality
from source.benchmarks.mediated_causality import causality_from_table
from source.benchmarks.mediated_causality import MediatedCausalityArithmetic
from source.benchmarks.mediated_causality import get_table

def test_causality_from_table_tdist():
    # Verifies the front-door criterion calculation
    exam = MediatedCausality(
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
    p_diff,p_diff_ci_lower,p_diff_ci_upper,_n = causality_from_table(pearl_table, 'tdist')
    assert p_diff == (0.4525 - 0.4975)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.08872766149704325,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.00127233850295672,3)

def test_causality_from_table_bootstrap():
    # Verifies the front-door criterion calculation
    exam = MediatedCausality(
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
    p_diff,p_diff_ci_lower,p_diff_ci_upper,_n = causality_from_table(pearl_table, 'bootstrap')
    assert np.round(p_diff,3) == np.round(0.4525 - 0.4975,3)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.05151729437221199,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.03808042079620048,3)

def test_arithmetic():
    """
    Parametrized test for various mediatedCausalityArithmetic exam variants:
    - No NaNs in p_diff
    - Correct output shapes
    - Even distribution of answers and difficulties
    """
    n_problems = 9

    exam = MediatedCausalityArithmetic(n_problems=n_problems)
    metadata = exam.get_metadata()
    solutions = exam.get_solutions()
    questions = exam.get_questions()

    # Check for NaNs in p_diff
    p_diff = np.array(metadata["p_diff"], dtype=float)
    assert not np.any(np.isnan(p_diff)), f"{'mediatedCausalityArithmetic'}: p_diff contains NaNs"

    # Check the question & answer pairs are numerically correct
    for i in range(0,n_problems):
        num = re.findall(r"\d+\.\d+", questions[i])
        num = [float(x) for x in num]
        ans = num[0] * (num[1] * num[2] + num[3] * num[4]) + num[5] * (num[6] * num[7] + num[8] * num[9]) - num[10] * (num[11] * num[12] + num[13] * num[14]) - num[15] * (num[16] * num[17] + num[18] * num[19])
        assert abs(float(solutions[i]) - ans) < 1e-4, "Values are not approximately equal"

    # Check output dimensions
    assert len(solutions) == n_problems
    assert len(questions) == n_problems
    assert len(metadata["p_diff"]) == n_problems
    assert len(metadata["difficulty"]) == n_problems

    # Check answer difficulty level counts
    for level in ["easy", "medm", "hard"]:
        count = np.count_nonzero(metadata["difficulty"] == level)
        assert count == n_problems // 3, f"{'mediatedCausalityArithmetic'}: {level} count incorrect ({count})"

    # Checks that all prompts are the same length 
    # (numbers are rounded so this is certain)
    for i in range(0,n_problems):
        str = questions[i]
        assert len(str) == 211, f"String length is {len(str)}, expected 211 for non-unicode x's"
    
@pytest.mark.parametrize("exam_name", [
    #"mediatedCausalitySmoking_tdist",
    #"mediatedCausality_tdist",
    "mediatedCausalityWithMethod_tdist",
    # "mediatedCausalitySmoking_bootstrap",
    # "mediatedCausalityWithMethod_bootstrap",
    # "mediatedCausality_bootstrap",
])
def test_prompts_nans_and_output_dims(exam_name):
    """
    Parametrized test for various mediatedCausality exam variants:
    - No NaNs in p_diff
    - Correct output shapes
    - Even distribution of answers and difficulties
    - Verify the prompt & answer match
    """
    n_problems = 9
    plot_path = "./figures/"

    exam = MediatedCausality(plot_path, exam_name, n_problems=n_problems, n_bootstrap=200)
    metadata = exam.get_metadata()
    solutions = exam.get_solutions()
    questions = exam.get_questions()

    # Verify the prompt & answer match
    xyz = get_table()
    for i in range(len(questions)):
        numbers = [int(num) for num in re.findall(r"\d+", questions[i])]
        if len(numbers) > 8:
            numbers = numbers[0:8]
        table = np.hstack((xyz,np.transpose(np.array([numbers]))))
        p_diff,p_diff_ci_lower,p_diff_ci_upper,_n = causality_from_table(table, 'tdist')
        assert np.round(p_diff,4) == np.round(metadata["p_diff"][i],4)

    # Check for NaNs in p_diff
    p_diff = np.array(metadata["p_diff"], dtype=float)
    assert not np.any(np.isnan(p_diff)), f"{exam_name}: p_diff contains NaNs"

    # Check output dimensions
    assert len(solutions) == n_problems
    assert len(questions) == n_problems
    assert len(metadata["p_diff"]) == n_problems
    assert len(metadata["p_diff_ci_upper"]) == n_problems
    assert len(metadata["p_diff_ci_lower"]) == n_problems
    assert len(metadata["n_samples"]) == n_problems
    assert len(metadata["difficulty"]) == n_problems

    # Check answer (A,B,C) and difficulty level counts
    for label in ["A", "B", "C"]:
        count = np.count_nonzero(solutions == label)
        assert count == n_problems // 3, f"{exam_name}: {label} count incorrect ({count})"

    for level in ["easy", "intermediate", "difficult"]:
        count = np.count_nonzero(metadata["difficulty"] == level)
        assert count == n_problems // 3, f"{exam_name}: {level} count incorrect ({count})"



