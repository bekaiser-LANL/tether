""" Randomly generate new benchmarks """
import os
import argparse
from source.generator import generate_benchmarks
from source.utils import get_parser

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    parser = get_parser(script="generate")
    args = parser.parse_args()

    generate_benchmarks(
        path=args.path,
        exam_name=args.exam_name,
        n_problems=args.n_problems,
        plot_flag=args.make_plots,
        exam_idx=args.exam_idx,
        n_numbers=args.n_numbers,
        verbose=args.verbose
    )
    
    # # Model is not needed for generation (it's needed for run.py):
    # model_path=args.model_path
    # if model_path:
    #     if not os.path.isdir(model_path):
    #         print(f"The directory '{model_path}' does not exist.")
    #     else:
    #         print(f"Using locally downloaded model")
    # else:
    #     print("Using API.")

    print(f"\n {args.exam_name} benchmark generated at {args.path}!\n")

if __name__ == "__main__":
    main()
