""" Analyze benchmark results """
import os
import argparse
from source.analyzer import Analyzer
from source.utils import detect_duplicate_tables, load_saved_benchmark

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the completed benchmark for a specified model."
    )

    parser.add_argument(
        "benchmark_name",
        help="Name of the benchmark to run, including its index (e.g., MediatedCausality_tdist_0)"
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

    # Analyzer(args.path, args.model_name, args.benchmark_name)
    # print(f"\n Analyses of benchmark '{args.benchmark_name}' for model '{args.model_name}' completed at: {args.path}")

    # Verify that each mediated causality benchmark has no duplicate problems:
    exam_idx = 0
    exam_names = ['MediatedCausality_bootstrap',
                  'MediatedCausalitySmoking_bootstrap',
                  'MediatedCausalityWithMethod_bootstrap',
                  'MediatedCausality_tdist',
                  'MediatedCausalitySmoking_tdist', # <- has a duplicate
                  'MediatedCausalityWithMethod_tdist'
                  ]
    for i in range(0,len(exam_names)):
        data = load_saved_benchmark(data_path + '/blank/',exam_names[i], exam_idx)
        has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(data['table'])
        print(f"\n Benchmark: {exam_names[i]}"
            f"\n Duplicate tables detected: {has_duplicates}"
            f"\n Number of problems: {n_problems}")
        if has_duplicates:
            print(f" {duplicate_pairs} duplicate pairs found")

    # Verify the blank standard deviation benchmark:
    exam_idx = 0
    exam_name = 'StandardDeviation'
    data = load_saved_benchmark(data_path + '/blank/',exam_name, exam_idx)
    n_problems = len(data["question"])
    for i in range(0,2):
        print('\n question = ',data["question"][i])
        print(' unbiased solution = ',data["unbiased_solution"][i])
        print(' biased solution = ',data["biased_solution"][i])

    # Verify a completed benchmark:
    exam_idx = 0
    exam_name = 'MediatedCausality_bootstrap_gpt-4.1'
    data = load_saved_benchmark(data_path + '/completed/',exam_name, exam_idx)
    n_problems = len(data["question"])
    for i in range(0,n_problems):
        print('\n question = ',data["question"][i])
        print(' responses = ',data["responses"][i])
        print(' solution = ',data["solution"][i])

if __name__ == "__main__":
    main()
