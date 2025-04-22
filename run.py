""" Run a previously generated benchmark on an LLM """
import sys
from source.proctor import Proctor
from cli_args import get_parser 

def main():
    """ Run the benchmark """
    parser = get_parser(script="run")
    args = parser.parse_args()
    kwargs = vars(args)
    #if len(sys.argv) < 4:
        # Usage: python run.py <benchmark_name>" "<model_name>
        #        <path_to_benchmarks>"
        # Example: python run.py MediatedCausality_tdist gpt-4o
        #        /Users/Desktop/benchmarks/
    #    sys.exit(1)

    #benchmark = sys.argv[1]
    #model = sys.argv[2]
    #path_to_benchmarks = sys.argv[3]
    benchmark = kwargs.pop("exam_name")
    model = kwargs.pop("model")
    path_to_benchmarks = kwargs.pop("path")
    #verbose = kwargs.pop("verbose", False)

    Proctor(benchmark, model, path_to_benchmarks, verbose=True, **kwargs)

    print(f"\n Benchmark {benchmark} of model {model} complete")

if __name__ == "__main__":
    main()
