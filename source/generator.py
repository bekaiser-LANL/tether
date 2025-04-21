""" Randomly generates and saves benchmarks as .npz files """
from .utils import create_missing_directory, SaveBenchmark
from .benchmarks.mediated_causality import MediatedCausality
from .benchmarks.standard_deviation import StandardDeviation
from .benchmarks.significant_figures import SignificantFigures
from .benchmarks.simpleInequality import simpleInequality

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
    n_numbers = kwargs.get('n_numbers',20)
    # index for repeated benchmarks:
    exam_idx = kwargs.get('exam_idx', 'unset')
    # number of problems in the benchmark
    n_problems = kwargs.get('n_problems', 180)
    # flag for plotting extra generated benchmark data:
    plot_flag = kwargs.get("plot_flag", False)
    # path to benchmark reports:
    results_path = path + 'results/'
    # save blank benchmarks for repeated use:
    save_path = path + 'saved/'
    # save figures path for extra generated benchmark data:
    plot_path = save_path + exam_name + '_figures/'

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
    elif exam_name_wo_ci_method == 'simpleInequality':

        # Generate all of the problems in the benchmark:
        problems = simpleInequality(
            n_numbers=n_numbers,
            n_problems=n_problems
        )

        # # Save the benchmark as an .npz
        # saver = SaveBenchmark.from_simple_inequality(
        #     source=problems,
        #     path=self.path,
        #     exam_name=self.exam_name,
        #     exam_idx=self.exam_idx
        # )

    elif exam_name_wo_ci_method == 'StandardDeviation':

        # Generate all of the problems in the benchmark:
        problems = StandardDeviation(
            n_numbers=n_numbers,
            n_problems=n_problems
        )

        # # Save the benchmark as an .npz
        # saver = SaveBenchmark.from_mediated_causality(
        #     source=problems,
        #     path=self.path,
        #     exam_name=self.exam_name,
        #     exam_idx=self.exam_idx
        # )

    elif exam_name_wo_ci_method in ('MediatedCausalitySmoking',
                                            'MediatedCausality'):

        # Generate all of the problems in the benchmark:
        problems = MediatedCausality(
            plot_path=plot_path,
            exam_name=exam_name,
            plot_flag=plot_flag,
            n_problems=n_problems,
        )

        # Save the benchmark as an .npz
        saver = SaveBenchmark.from_mediated_causality(
            source=problems,
            path=save_path,
            exam_name=exam_name,
            exam_idx=exam_idx
        )

    #saver.save_attributes()
