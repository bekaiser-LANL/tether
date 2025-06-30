""" Tests functions and classes in utils """
import os
import tempfile
import numpy as np
from source import utils  # Adjust this if utils.py is elsewhere

def test_strip_after_second_underscore():
    """ Test string manipulation """
    assert utils.strip_after_second_underscore("A_B_C") == "A_B"
    assert utils.strip_after_second_underscore("OnlyOne") == "OnlyOne"

def test_get_after_second_underscore():
    """ Test string manipulation """    
    assert utils.get_after_second_underscore("A_B_C") == "C"
    assert utils.get_after_second_underscore("No_Underscore") == ""

def test_get_npz_filename():
    """ Test string manipulation """    
    path = "/tmp"
    name = "Benchmark"
    model = "gpt"
    idx = "1"
    expected = os.path.join(path, "Benchmark_gpt_1.npz")
    assert utils.get_npz_filename(path, name, idx, model) == expected

def test_get_npz_filename_no_model():
    """ Test string manipulation """      
    path = "/tmp"
    name = "Benchmark"
    idx = "1"
    expected = os.path.join(path, "Benchmark_1.npz")
    assert utils.get_npz_filename_no_model(path, name, idx) == expected

def test_is_integer():
    """ Test is integer """      
    assert utils.is_integer(5)
    assert not utils.is_integer("5")

def test_standard_error_for_proportion():
    """ Test is SE for proportion """       
    result = utils.standard_error_for_proportion(0.5, 100)
    assert np.isclose(result, 0.05, atol=1e-3)

def test_get_95_CI_tdist():
    """ Test t distribution to estimate for 95% CI """
    upper, lower = utils.get_95_CI_tdist(0.5, 100)
    assert upper > lower

def test_enforce_probability_bounds():
    """ Test probability bound enforcement """
    assert utils.enforce_probability_bounds(1.2) == 1.0
    assert utils.enforce_probability_bounds(-0.5) == 0.0
    assert utils.enforce_probability_bounds(0.5) == 0.5

def test_create_missing_directory():
    """ Test directory creation """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = os.path.join(tmpdir, "new_dir")
        utils.create_missing_directory(test_path)
        assert os.path.exists(test_path)

def test_is_divisible_by_9_and_3():
    """ Test multiples of 3 and 9 """
    assert utils.is_divisible_by_9(18)
    assert not utils.is_divisible_by_9(20)
    assert utils.is_divisible_by_3(9)
    assert not utils.is_divisible_by_3(10)
