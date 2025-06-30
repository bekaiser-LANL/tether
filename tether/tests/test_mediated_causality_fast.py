""" Fast Tests for Mediated Causality """
import numpy as np
import pytest
from source.benchmarks.mediated_causality import causality_from_table
from source.benchmarks.mediated_causality import causality_from_frequency
from source.benchmarks.mediated_causality import generate_dataset_by_difficulty

# def causality_from_frequency(array):
#     n000,n001,n010,n011,n100,n101,n110,n111 = array
#     A = n010*(n000+n010+n001+n011)/(n000+n010)+n110*(n100+n110+n101+n111)/(n100+n110)
#     B = n011*(n000+n010+n001+n011)/(n001+n011)+n111*(n100+n110+n101+n111)/(n101+n111)
#     n = np.sum(array)
#     PdoX1 = ( (n110+n100)*A + (n111+n101)*B ) / (n*(n111+n101+n110+n100))
#     PdoX0 = ( (n010+n000)*A + (n011+n001)*B ) / (n*(n011+n001+n010+n000))
#     dP = PdoX1-PdoX0 # > 0 causal, <= 0 not causal
#     return dP

def test_causality_from_table_tdist():
    """ Verifies the front-door criterion calculation """
    # This table is equivalent to the table on p.84 of Pearl "Causality":
    pearl_table = np.array([[0,0,0,427.50],
                            [0,0,1, 23.75],
                            [0,1,0, 47.50],
                            [0,1,1,  1.25],
                            [1,0,0,  2.50],
                            [1,0,1, 71.25],
                            [1,1,0, 22.50],
                            [1,1,1,403.75]])
    result = causality_from_table(pearl_table, 'tdist')
    p_diff, p_diff_ci_lower, p_diff_ci_upper = result[:3]
    assert p_diff == (0.4525 - 0.4975)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.08872766149704325,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.00127233850295672,3)

def test_causality_from_table_bootstrap():
    """ Verifies the front-door criterion calculation """
    # This table is equivalent to the table on p.84 of Pearl "Causality":
    pearl_table = np.array([[0,0,0,42750],
                            [0,0,1, 2375],
                            [0,1,0, 4750],
                            [0,1,1,  125],
                            [1,0,0,  250],
                            [1,0,1, 7125],
                            [1,1,0, 2250],
                            [1,1,1,40375]])
    result = causality_from_table(pearl_table, 'bootstrap', n_bootstrap=1000)
    p_diff, p_diff_ci_lower, p_diff_ci_upper = result[:3]
    assert np.round(p_diff,3) == np.round(0.4525 - 0.4975,3)
    assert np.round(p_diff_ci_lower,3) == np.round(-0.05151729437221199,3)
    assert np.round(p_diff_ci_upper,3) == np.round(-0.03808042079620048,3)

def test_causality_known_input():
    # Simple known input where causality is easy to compute
    array = np.array([10, 10, 10, 10, 10, 10, 10, 10])
    dP = causality_from_frequency(array)
    assert isinstance(dP, float), "Output should be a float"
    assert np.isfinite(dP), "Output should be a finite number"

def test_causality_random_input():
    # Random positive numbers
    array = np.random.randint(1, 100, size=8)
    dP = causality_from_frequency(array)
    assert isinstance(dP, float)
    assert np.isfinite(dP)

def test_causality_aginst_equation_input():
    # Random positive numbers
    array = np.array([285., 97., 216., 200., 211., 57., 94., 97.])
    dP1 = causality_from_frequency(array)
    n000,n001,n010,n011,n100,n101,n110,n111 = array
    A1 = n010*(n000+n010+n001+n011)/(n000+n010)
    A2 = n110*(n100+n110+n101+n111)/(n100+n110)
    A = A1 + A2    
    B1 = n011*(n000+n010+n001+n011)/(n001+n011)
    B2 = n111*(n100+n110+n101+n111)/(n101+n111)
    B = B1 + B2    
    n = np.sum(array)
    PdoX1 = ( (n110+n100)*A + (n111+n101)*B ) / (n*(n111+n101+n110+n100))
    PdoX0 = ( (n010+n000)*A + (n011+n001)*B ) / (n*(n011+n001+n010+n000))
    dP2 = PdoX1-PdoX0 # 
    assert dP1 == dP2

def test_causality_zero_input():
    # Example array that would cause div by zero
    array = np.array([0, 0, 0, 1, 0, 0, 0, 1])  
    dP = causality_from_frequency(array)
    assert np.isnan(dP), "Should return np.nan when division by zero would occur"


def test_generate_easy():
    difficulty_threshold = np.array([0.05, 0.25])
    factor_range = np.linspace(10, 20, 5)
    result = generate_dataset_by_difficulty('easy', difficulty_threshold, factor_range)
    assert result.shape == (8,), "Result should have 8 elements"
    assert np.all((np.isnan(result)) | (result >= 1)), "All values should be >= 1 or NaN"

def test_generate_hard():
    difficulty_threshold = np.array([0.05, 0.25])
    factor_range = np.linspace(10, 20, 5)
    result = generate_dataset_by_difficulty('hard', difficulty_threshold, factor_range)
    assert result.shape == (8,), "Result should have 8 elements"
    assert np.all((np.isnan(result)) | (result >= 1)), "All values should be >= 1 or NaN"

def test_generate_medium():
    difficulty_threshold = np.array([0.05, 0.25])
    factor_range = np.linspace(10, 20, 5)
    result = generate_dataset_by_difficulty('medium', difficulty_threshold, factor_range)
    assert result.shape == (8,), "Result should have 8 elements"
    assert np.all((np.isnan(result)) | (result >= 1)), "All values should be >= 1 or NaN"

def test_generate_invalid_difficulty():
    difficulty_threshold = np.array([0.05, 0.25])
    factor_range = np.linspace(10, 20, 5)
    # Should probably not crash even if difficulty is unknown
    with pytest.raises(Exception):
        generate_dataset_by_difficulty('unknown', difficulty_threshold, factor_range)


# def test_arithmetic():
#     """
#     Parametrized test for various mediatedCausalityArithmetic exam variants:
#     - No NaNs in p_diff
#     - Correct output shapes
#     - Even distribution of answers and difficulties
#     """
#     n_problems = 9

#     exam = MediatedCausalityArithmetic(n_problems=n_problems)
#     metadata = exam.get_metadata()
#     solutions = exam.get_solutions()
#     questions = exam.get_questions()

#     # Check for NaNs in p_diff
#     p_diff = np.array(metadata["p_diff"], dtype=float)
#     assert not np.any(np.isnan(p_diff))

#     # Check the question & answer pairs are numerically correct
#     for i in range(0,n_problems):
#         num = re.findall(r"\d+\.\d+", questions[i])
#         num = [float(x) for x in num]
#         a = num[0] * (num[1] * num[2] + num[3] * num[4])
#         b = num[5] * (num[6] * num[7] + num[8] * num[9])
#         c = num[10] * (num[11] * num[12] + num[13] * num[14])
#         d = num[15] * (num[16] * num[17] + num[18] * num[19])
#         ans = a + b - c - d
#         assert float(solutions[i]) == approx(ans, abs=1e-4)

#     # Check output dimensions
#     assert len(solutions) == n_problems
#     assert len(questions) == n_problems
#     assert len(metadata["p_diff"]) == n_problems
#     assert len(metadata["difficulty"]) == n_problems

#     # Check answer difficulty level counts
#     for level in ["easy", "medm", "hard"]:
#         count = np.count_nonzero(metadata["difficulty"] == level)
#         assert count == n_problems // 3

#     # Checks that all prompts are the same length (211 characters)
#     # (numbers are rounded so this is certain)
#     for i in range(0,n_problems):
#         str = questions[i]
#         assert len(str) == 211
