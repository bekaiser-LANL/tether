""" Randomly generate new benchmarks """
import argparse

def get_parser(script="generate"):
    parser = argparse.ArgumentParser(
        description="Generate a benchmark dataset."
    )

    if script == "run":
        parser.add_argument(
            "exam_name",
            help="Name of the benchmark to generate"
        )
        parser.add_argument(
            "model",
            help="Name of model to use"
        )
        parser.add_argument(
            "path",
            help="Directory path to /benchmarks/"
        )
    if script == "generate":
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
        help="Optional path for locally downloaded model"
    )
    return parser
