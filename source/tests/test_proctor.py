""" Tests for Proctor """
from unittest import mock
import pytest
import numpy as np
from source.proctor import Proctor
# pylint: disable=redefined-outer-name

@pytest.fixture
def sample_benchmark():
    """ Sample benchmark """
    return {
        "question": np.array(["What is 2+2?", "What is the capital of France?"]),
        "difficulty": np.array(["easy", "medium"]),
        "name": np.array(["Math", "Geo"])
    }

@mock.patch("source.proctor.create_missing_directory")
@mock.patch("source.proctor.load_saved_benchmark")
@mock.patch("source.proctor.Proctor.give_benchmark")
@mock.patch("source.proctor.np.savez")
def test_proctor_init(
    mock_savez,
    mock_give_benchmark,
    mock_load_benchmark,
    sample_benchmark
    ):
    """ Test of Proctor initialization """
    mock_load_benchmark.return_value = sample_benchmark
    mock_give_benchmark.return_value = np.array(["4", "Paris"])

    proctor = Proctor("path/to/benchmarks", "gpt-4o", "TestExam", verbose=True)

    assert proctor.exam_name == "TestExam"
    assert proctor.model == "gpt-4o"
    assert mock_savez.called

def test_ask_openai_success():
    """ Test if openai API can be opened"""
    mock_client = mock.Mock()
    mock_response = mock.Mock()
    mock_response.choices = [mock.Mock(message=mock.Mock(content="This is a test"))]
    mock_client.chat.completions.create.return_value = mock_response

    proctor = Proctor.__new__(Proctor)  # Bypass __init__
    proctor.client = mock_client
    proctor.temperature = 0.0

    result = proctor.ask_openai("What's up?", "gpt-4o")
    assert result == "This is a test"

@mock.patch("source.proctor.Proctor.give_question_to_llm")
def test_give_benchmark(mock_give_q, sample_benchmark):
    """ Test of function: test_give_benchmark """
    mock_give_q.side_effect = ["4", "Paris"]

    proctor = Proctor.__new__(Proctor)
    proctor.verbose = False
    proctor.model = "gpt-4o"

    responses = proctor.give_benchmark(sample_benchmark)
    assert isinstance(responses, np.ndarray)
    assert responses.tolist() == ["4", "Paris"]

# @mock.patch("requests.post")
# def test_give_question_to_llama(mock_post):
#     """ Test of function: test_give_question_to_llm_llama """
#     mock_post.return_value.status_code = 200
#     mock_post.return_value.json.return_value = {"response": "Test response"}

#     proctor = Proctor.__new__(Proctor)
#     proctor.model = "llama3"

#     result = proctor.give_question_to_llm("Prompt?")
#     assert result == "Test response"
