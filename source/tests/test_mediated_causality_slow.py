""" Slow Tests for Mediated Causality """
import re
import pytest
import numpy as np
from source.benchmarks.mediated_causality import MediatedCausality
from source.benchmarks.mediated_causality import causality_from_table
from source.benchmarks.mediated_causality import get_table

EXAM_NAMES = [
    "MediatedCausalitySmoking_tdist",
    # "MediatedCausality_tdist",
    # "MediatedCausalityWithMethod_tdist",
    # "MediatedCausalitySmoking_bootstrap",
    # "MediatedCausalityWithMethod_bootstrap",
    # "MediatedCausality_bootstrap",
]

def get_method(name: str) -> str:
    if name.endswith("_tdist"):
        return "tdist"
    elif name.endswith("_bootstrap"):
        return "bootstrap"
    raise ValueError(f"Unknown method in name: {name}")

def get_prefix(s: str) -> str:
    return s.split('_', 1)[0]

@pytest.mark.parametrize("exam_name", EXAM_NAMES)
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
    p_diff = exam.get_p_diff()
    solutions = np.array(exam.get_solutions())
    questions = exam.get_questions()

    # Verify the prompt & answer match
    xyz = get_table()
    for i in range(len(questions)):
        numbers = [int(num) for num in re.findall(r"\d+", questions[i])]
        if len(numbers) > 8:
            numbers = numbers[0:8]
        table = np.hstack((xyz,np.transpose(np.array([numbers]))))
        result = causality_from_table(table, get_method(exam_name))
        p_diff_verify = result[:1]
        assert np.allclose(p_diff[i], p_diff_verify, atol=1e-4)


    # Check for NaNs in p_diff
    #p_diff = np.array(metadata["p_diff"], dtype=float)
    assert not np.any(np.isnan(p_diff))

    # Check output dimensions
    assert len(solutions) == n_problems
    assert len(questions) == n_problems
    assert len(p_diff) == n_problems
    assert len(exam.get_p_diff_ci_upper()) == n_problems
    assert len(exam.get_p_diff_ci_lower()) == n_problems
    assert len(exam.get_n_samples()) == n_problems
    assert len(exam.get_difficulty()) == n_problems

    # Check answer (A,B,C) and difficulty level counts
    for label in ["A", "B", "C"]:
        count = np.count_nonzero(solutions == label)
        assert count == n_problems // 3

    for level in ["easy", "medium", "hard"]:
        count = np.count_nonzero(np.array(exam.get_difficulty()) == level)
        assert count == n_problems // 3
