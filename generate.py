""" Randomly generate new benchmarks """
import argparse
from source.generator import generate_benchmarks

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
        help="Directory path to /benchmarks/"
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
    parser.add_argument(
        "--model_path",
        type=str,
        help="path for locally downloaded model"
    )

    args = parser.parse_args()

    generate_benchmarks(
        path=args.path,
        exam_name=args.exam_name,
        n_problems=args.n_problems,
        plot_flag=args.make_plots,
        exam_idx=args.exam_idx,
        n_numbers=args.n_numbers,
        model_path=args.model_path
    )
    
    model_path=args.model_path
    if model_path:
        if not os.path.isdir(directory):
            print(f"The directory '{directory}' does not exist.")
        else:
            print(f"Using locally downloaded model")
    else:
        print("Using API.")

    print(f"\n {args.exam_name} benchmark generated at {args.path}!\n")

if __name__ == "__main__":
    main()
