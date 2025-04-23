""" Run a previously generated benchmark on an LLM """
import os
import argparse
from source.proctor import Proctor

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    """ Run the benchmark """
    parser = get_parser(script="run")
    args = parser.parse_args()
    kwargs = vars(args)
    benchmark = kwargs.pop("exam_name")
    model = kwargs.pop("model")
    path_to_benchmarks = kwargs.pop("path")
    #verbose = kwargs.pop("verbose", False)

    Proctor(benchmark, model, path_to_benchmarks, verbose=True, **kwargs)

    print(f"\n Benchmark '{args.benchmark_name}' for model '{args.model_name}' completed at: {args.path}")

if __name__ == "__main__":
    main()
