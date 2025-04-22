""" Randomly generate new benchmarks """
import os
import argparse
from source.generator import generate_benchmarks

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    """ Generate the benchmark """
    parser = argparse.ArgumentParser(
        description="Generate a benchmark dataset."
    )
    parser.add_argument(
        "exam_name",
        help="Name of the benchmark to generate"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=data_path,
        help=f"Directory path to /benchmarks/ (default: {data_path})"
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=180,
        help="Number of problems to generate for the benchmark"
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        help="Enable plotting"
    )
    parser.add_argument(
        "--n_numbers",
        type=int,
        default=20,
        help="Number of integers for standard deviation benchmark"
    )
    parser.add_argument(
        "--exam_idx",
        type=int,
        default=0,
        help="Index for multiple benchmarks of the same type"
    )

    args = parser.parse_args()

    generate_benchmarks(
        path=args.path,
        exam_name=args.exam_name,
        n_problems=args.n_problems,
        plot_flag=args.make_plots,
        exam_idx=args.exam_idx,
        n_numbers=args.n_numbers
    )

    print(f"\n {args.exam_name} benchmark generated at {args.path}!\n")

if __name__ == "__main__":
    main()
