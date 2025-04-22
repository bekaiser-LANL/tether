""" Randomly generate new benchmarks """
from cli_args import get_parser
import os
from source.generator import generate_benchmarks


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
        model_path=args.model_path
    )
    
    model_path=args.model_path
    if model_path:
        if not os.path.isdir(model_path):
            print(f"The directory '{model_path}' does not exist.")
        else:
            print(f"Using locally downloaded model")
    else:
        print("Using API.")

    print(f"\n {args.exam_name} benchmark generated at {args.path}!\n")

if __name__ == "__main__":
    main()
