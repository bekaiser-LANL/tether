""" Run a previously generated benchmark on an LLM """
import os
import argparse
from source.proctor import Proctor

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    parser = argparse.ArgumentParser(
        description="Run a benchmark on a specified model."
    )

    parser.add_argument(
        "benchmark_name",
        help="Name of the benchmark to run (e.g., MediatedCausality_tdist)"
    )

    parser.add_argument(
        "model_name",
        help="Name of the model to test (e.g., gpt-4o)"
    )

    parser.add_argument(
        "--path",
        default=data_path,
        help=f"Path to the benchmarks directory (default: from PATH_TO_BENCHMARKS or '{data_path}')"
    )

    args = parser.parse_args()

    Proctor(args.path, args.model_name, args.benchmark_name, verbose=True)

    print(f"\n Benchmark '{args.benchmark_name}' for model '{args.model_name}' completed at: {args.path}")

if __name__ == "__main__":
    main()
