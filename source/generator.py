""" Randomly generates and saves benchmarks as .npz files """
import numpy as np
from .utils import create_missing_directory, SaveBenchmark
from .benchmarks.mediated_causality import MediatedCausality
from .benchmarks.standard_deviation import StandardDeviation
from .benchmarks.significant_figures import SignificantFigures

class Generator():
    """ Randomly generates and saves benchmarks as .npz files """

    def __init__(self, path, exam_name, **kwargs):

        self.exam_name = exam_name
        self.path = path # dir for benchmark .npz, plots
        
        # Checkpoint frequency if an integer,
        # no checkpoint .npz output if a NaN:
        self.checkpoint_freq = kwargs.get('checkpoint_freq', 'unset')
        # Restart question number if an integer, start at question 1 if a NaN:
        self.restart_idx = kwargs.get('restart_idx', 'unset')
        # for OpenAI reasoning models only:
        self.reasoning_effort = kwargs.get('reasoning_effort', 'high')
        # for OpenAI non-reasoning models only:
        self.temperature  = kwargs.get('temperature', 0.0)
        # save blank benchmark as .txt:
        self.record_txt = kwargs.get('record_txt', False)
        # number of numbers for standard deviation benchmark:
        self.n_numbers = kwargs.get('n_numbers',20)
        # index for repeated benchmarks:
        self.exam_idx = kwargs.get('exam_idx', 'unset')
        # number of problems in the benchmark
        self.n_problems = kwargs.get('n_problems', 360)
        # flag for plotting extra generated benchmark data:
        self.plot_flag = kwargs.get("plot_flag", False)
        # path to benchmark reports:
        self.results_path = self.path + 'results/'
        # save blank benchmarks for repeated use:
        self.save_path = self.path + 'saved/'
        # save figures path for extra generated benchmark data:
        self.plot_path = self.save_path + exam_name + '_figures/'

        create_missing_directory(self.path)
        create_missing_directory(self.save_path)
        create_missing_directory(self.results_path)
        create_missing_directory(self.plot_path)
 
        if '_' in self.exam_name:
            self.ci_method = (self.exam_name).split('_')[1]
            self.exam_name_wo_ci_method = (self.exam_name).split('_')[0]
        else:
            self.exam_name_wo_ci_method = self.exam_name

        if self.exam_name_wo_ci_method == 'SignificantFigures':

            # Generate all of the problems in the benchmark:
            self.problems = SignificantFigures(
                n_problems=self.n_problems
            )

            # Save the benchmark as an .npz

        elif self.exam_name_wo_ci_method == 'StandardDeviation':

            # Generate all of the problems in the benchmark:
            self.problems = StandardDeviation(
                n_numbers=self.n_numbers,
                n_problems=self.n_problems
            )

        elif self.exam_name_wo_ci_method in ('MediatedCausalitySmoking',
                                             'MediatedCausality'):

            # Generate all of the problems in the benchmark:
            problems = MediatedCausality(
                plot_path=self.plot_path,
                exam_name=self.exam_name,
                plot_flag=self.plot_flag,
                n_problems=self.n_problems,
            )

            # Save the benchmark as an .npz
            saver = SaveBenchmark.from_mediated_causality(
                source=problems,
                path=self.path,
                exam_name=self.exam_name,
                exam_idx=self.exam_idx
            )
        
        saver.save_attributes()

        # data = np.load("/Users/l281800/Desktop/benchmarks/saved/MediatedCausalitySmoking_tdist.npz", allow_pickle=True)
        # # pickle is needed to load a dict
        # print(data.files) 
        # #print(np.shape(data['table']))
        # print(data['difficulty'])
        # print(data['p_diff'])
        # #print(data['name'])
        # print(data['n_samples'])