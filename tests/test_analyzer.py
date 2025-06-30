""" Test the script that runs (calls Proctor) benchmarks on LLMs """
from tether.core.analyzer import extract_boolean_result_from_response, truncate_response


def test_truncate_response_short():
    text = "Line1\nLine2\nLine3"
    result = truncate_response(text)
    assert result == text


def test_truncate_response_long():
    text = "\n".join([f"Line{i}" for i in range(10)])
    result = truncate_response(text, num_start=2, num_end=2)
    assert "Line0" in result and "Line9" in result
    assert "... (omitted middle lines) ..." in result


def test_extract_boolean_result_from_response_valid_true():
    response = """```json
    {
      "result": true,
      "explanation": "It matches the solution."
    }
    ```"""
    assert extract_boolean_result_from_response(response) is True


def test_extract_boolean_result_from_response_valid_false():
    response = """```json
    {
      "result": false,
      "explanation": "Mismatch."
    }
    ```"""
    assert extract_boolean_result_from_response(response) is False


def test_extract_boolean_result_from_response_invalid():
    response = "Not a json block"
    assert extract_boolean_result_from_response(response) is None
