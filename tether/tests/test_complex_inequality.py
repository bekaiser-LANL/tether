import numpy as np
import pytest
from scipy.integrate import trapezoid
from source.benchmarks.complex_inequality import ComplexInequality  # adjust as needed

@pytest.fixture
def ci_instance():
    return ComplexInequality(plot_path=".", exam_name="ComplexInequality_tdist", n_numbers=100, generate_flag=False)

def test_generate_vector_output_shape(ci_instance):
    vec, xvec, yvec = ci_instance.generate_vector(-10, 10, 1000)
    assert len(vec) == ci_instance.n_numbers
    assert len(xvec) == 1000
    assert len(yvec) == 1000

def test_generate_vector_pair_outputs(ci_instance):
    vec1, vec2, std1, std2, x1, y1, x2, y2 = ci_instance.generate_vector_pair((0.5, 1.0))
    assert len(vec1) == ci_instance.n_numbers
    assert len(vec2) == ci_instance.n_numbers
    assert isinstance(std1, float)
    assert isinstance(std2, float)

def test_multimodal_pdf_area(ci_instance):
    x = np.linspace(-15, 15, 1000)
    y = ci_instance.multimodal_pdf(x)
    area = trapezoid(y, x)
    assert np.isclose(area, 1.0, atol=1e-2), f"PDF area is not 1 (area = {area})"

def test_calculate_ci(ci_instance):
    vec1 = np.random.normal(0, 1, ci_instance.n_numbers)
    vec2 = np.random.normal(1, 1, ci_instance.n_numbers)
    std1 = np.std(vec1)
    std2 = np.std(vec2)
    ci_lower, ci_upper, diff = ci_instance.calculate_ci(vec1, vec2, std1, std2)
    assert ci_upper > ci_lower
    assert isinstance(diff, float)

def test_bootstrap_ci(ci_instance):
    vec1 = np.random.normal(0, 1, ci_instance.n_numbers)
    vec2 = np.random.normal(1, 1, ci_instance.n_numbers)
    ci_lower, ci_upper, diff = ci_instance.bootstrap_ci(vec1, vec2)
    assert ci_upper > ci_lower
    assert isinstance(diff, float)

def test_record_solutions_A(ci_instance):
    _, answer = ci_instance.record_solutions(1.0, 2.0)
    assert answer == 'A'

def test_record_solutions_B(ci_instance):
    _, answer = ci_instance.record_solutions(-2.0, -1.0)
    assert answer == 'B'

def test_record_solutions_C(ci_instance):
    _, answer = ci_instance.record_solutions(-1.0, 1.0)
    assert answer == 'C'

def test_find_mode_difference(ci_instance):
    vec1 = np.array([1, 1, 1, 2, 3])
    vec2 = np.array([0, 0, 1, 2])
    mode1, mode2, diff = ci_instance.find_mode_difference(vec1, vec2)
    assert mode1 == 1
    assert mode2 == 0
    assert diff == 1

def test_assign_difficulty(ci_instance):
    assert ci_instance.assign_difficulty(0.3) == 'hard'
    assert ci_instance.assign_difficulty(1.0) == 'medium'
    assert ci_instance.assign_difficulty(2.0) == 'easy'
