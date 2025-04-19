""" Randomly generate new benchmarks """
import argparse
from source.generator import Generator

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
        help="Directory path to save the benchmark"
    )
    parser.add_argument(
        "--n_problems",
        type=int,
        default=180,
        help="Number of problems to generate"
    )
    parser.add_argument(
        "--plot_flag",
        action="store_true",
        help="Enable plotting"
    )

    args = parser.parse_args()

    Generator(
        path=args.path,
        exam_name=args.exam_name,
        n_problems=args.n_problems,
        plot_flag=args.plot_flag
    )

    print(f"\n {args.exam_name} benchmark generated at {args.path}!\n")

if __name__ == "__main__":
    main()
