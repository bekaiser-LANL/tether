""" Analyze benchmark results """
import os
import argparse
from source.analyzer import Analyzer
#from source.utils import detect_duplicate_tables, load_saved_benchmark
from source.utils import get_parser

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    """ Analyze the benchmark """
    parser = get_parser(script="analyze")
    args = parser.parse_args()
    kwargs = vars(args)
    npz_filename = kwargs.pop("exam_name")
    verbose = kwargs.pop("verbose", False)

    Analyzer(npz_filename, verbose=verbose, **kwargs)

if __name__ == "__main__":
    main()
