""" Run a previously generated benchmark on an LLM """
import sys
from source.proctor import Proctor
 

def main():
    """ Run the benchmark """
    print("we're running")
    if len(sys.argv) < 4:
        # Usage: python run.py <benchmark_name>" "<model_name>
        #        <path_to_benchmarks>"
        # Example: python run.py MediatedCausality_tdist gpt-4o
        #        /Users/Desktop/benchmarks/
        sys.exit(1)

    benchmark = sys.argv[1]
    model = sys.argv[2]
    path_to_benchmarks = sys.argv[3]

    Proctor(path_to_benchmarks, model, benchmark, verbose=True)

    print(f"\n Benchmark {benchmark} of model {model} complete")

if __name__ == "__main__":
    main()
