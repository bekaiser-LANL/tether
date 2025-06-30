""" Randomly generates and saves benchmarks as .npz files """
import os
from tether.core.utils import create_missing_directory, SaveBenchmark
from tether.benchmarks.mediated_causality import MediatedCausality
from tether.benchmarks.standard_deviation import StandardDeviation
from tether.benchmarks.simple_inequality import SimpleInequality
from tether.benchmarks.complex_inequality import ComplexInequality
from tether.core.utils import get_parser

DATA_PATH = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def generate_benchmarks(path, exam_name, **kwargs):
    """ Randomly generates and saves benchmarks as .npz files """
    # pylint: disable=too-many-instance-attributes

    # MOVE TO RUN:
    # Checkpoint frequency if an integer,
    # no checkpoint .npz output if a NaN:
    # checkpoint_freq = kwargs.get('checkpoint_freq', 'unset')
    # # Restart question number if an integer, start at question 1 if a NaN:
    # restart_idx = kwargs.get('restart_idx', 'unset')
    # # for OpenAI reasoning models only:
    # reasoning_effort = kwargs.get('reasoning_effort', 'high')
    # # for OpenAI non-reasoning models only:
    # temperature  = kwargs.get('temperature', 0.0)
    # # save blank benchmark as .txt:
    # record_txt = kwargs.get('record_txt', False)

    # number of numbers for standard deviation benchmark:
    n_numbers = kwargs.get('n_numbers',10)
    # index for repeated benchmarks:
    exam_idx = kwargs.get('exam_idx', 'unset')
    # flag for plotting extra generated benchmark data:
    plot_flag = kwargs.get("plot_flag", False)
    # path to benchmark reports:
    results_path = os.path.join(path, 'completed')
    # save blank benchmarks for repeated use:
    save_path = os.path.join(path, 'blank')
    # save figures path for extra generated benchmark data:
    plot_path = os.path.join(save_path, f"{exam_name}_figures")
    # terminal output:
    verbose = kwargs.get("verbose", False)
    # number of problems in the benchmark
    if exam_name == 'StandardDeviation':
        n_problems = kwargs.get('n_problems', 100)
    else:
        n_problems = kwargs.get('n_problems', 180)

    create_missing_directory(path)
    create_missing_directory(save_path)
    create_missing_directory(results_path)
    create_missing_directory(plot_path)

    if '_' in exam_name:
        exam_name_wo_ci_method = (exam_name).split('_')[0]
    else:
        exam_name_wo_ci_method = exam_name

    saver = None
    
    if exam_name_wo_ci_method == 'SignificantFigures':

        # Generate all of the problems in the benchmark:
        problems = SignificantFigures(
            n_problems=n_problems
        )

        # # Save the benchmark as an .npz
        # saver = SaveBenchmark.from_mediated_causality(
        #     source=problems,
        #     path=self.path,
        #     exam_name=self.exam_name,
        #     exam_idx=self.exam_idx
        # )

    elif exam_name_wo_ci_method in ('SimpleInequality', 'SimpleInequalityAgent', 'SimpleInequalityWithMethod'):

        # Generate all of the problems in the benchmark:
        problems = SimpleInequality(
            n_numbers=n_numbers,
            plot_path=plot_path,
            n_problems=n_problems,
            plot_flag=plot_flag,
            exam_name=exam_name
        )

        # Save the benchmark as an .npz
        saver = SaveBenchmark.from_simple_inequality(
             source=problems,
             path=save_path,
             exam_name=exam_name,
             exam_idx=exam_idx
        )

    elif exam_name_wo_ci_method in ('ComplexInequality', 'ComplexInequalityWithMethod'):

        # Generate all of the problems in the benchmark:
        problems = ComplexInequality(
            n_numbers=n_numbers,
            plot_path=plot_path,
            n_problems=n_problems,
            plot_flag=plot_flag,
            exam_name=exam_name
        )

        # Save the benchmark as an .npz
        saver = SaveBenchmark.from_complex_inequality(
             source=problems,
             path=save_path,
             exam_name=exam_name,
             exam_idx=exam_idx
        )

    elif exam_name_wo_ci_method == 'StandardDeviation':

        # Generate all of the problems in the benchmark:
        problems = StandardDeviation(
            exam_idx=exam_idx,
            n_numbers=n_numbers,
            n_problems=n_problems,
            exam_name=exam_name
        )

        # Save the benchmark as an .npz
        saver = SaveBenchmark.from_standard_deviation(
            source=problems,
            path=save_path,
            exam_name=exam_name,
            exam_idx=exam_idx
        )

    elif exam_name_wo_ci_method in ('MediatedCausalitySmoking',
                                    'MediatedCausality',
                                    'MediatedCausalityWithMethod'):

        # Generate all of the problems in the benchmark:
        problems = MediatedCausality(
            plot_path=plot_path,
            exam_name=exam_name,
            plot_flag=plot_flag,
            n_problems=n_problems,
            verbose=verbose,
        )

        # Save the benchmark as an .npz
        saver = SaveBenchmark.from_mediated_causality(
            source=problems,
            path=save_path,
            exam_name=exam_name,
            exam_idx=exam_idx
        )

    saver.save_attributes()

def main():
    """ Generate benchmarks """
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
