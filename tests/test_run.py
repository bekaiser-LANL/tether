""" Test the script that runs (calls Proctor) benchmarks on LLMs """
from unittest import mock
from tether.core.run import main

def test_run_main(monkeypatch):
    # Simulate CLI args like: python run.py /my/path BenchmarkName ModelName --n_problems 10
    monkeypatch.setattr("sys.argv", [
        "run.py",
        "/mock/path",
        "MockBenchmark",
        "MockModel",
        "--n_problems", "10",
        "--verbose"
    ])

    # Patch the Proctor class to avoid real execution
    with mock.patch("tether.core.run.Proctor") as MockProctor:
        main()
        
        # Check that Proctor was called with expected arguments
        MockProctor.assert_called_once()
        args, kwargs = MockProctor.call_args

        assert args[0] == "MockBenchmark"
        assert args[1] == "MockModel"
        assert args[2] == "/mock/path"
        assert kwargs["n_problems"] == 10
        assert kwargs["verbose"] is True
