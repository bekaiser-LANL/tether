""" Randomly generates and saves benchmarks as .npz files """
import numpy as np
from .recorder import RecordBenchmark
from .utils import create_missing_directory
from .benchmarks.mediated_causality import MediatedCausality
from .benchmarks.standard_deviation import StandardDeviation
from .benchmarks.significant_figures import SignificantFigures

class Generator():
    """ Randomly generates and saves benchmarks as .npz files """

    def __init__(self, path, exam_name, **kwargs):

        self.exam_name = exam_name
        self.path = path

        # Checkpoint frequency if an integer,
        # no checkpoint .npz output if a NaN:
        self.checkpoints = kwargs.get('checkpoints', np.nan)
        # Restart question number if an integer, start at question 1 if a NaN:
        self.restart = kwargs.get('restart', np.nan)
        # for OpenAI reasoning models only:
        self.reasoning_effort = kwargs.get('reasoning_effort', 'high')
        # for OpenAI non-reasoning models only:
        self.temperature  = kwargs.get('temperature', 0.0)
        # save blank benchmark as .txt:
        self.record_txt = kwargs.get('record_txt', False)
        # number of numbers for standard deviation benchmark:
        self.n_numbers = kwargs.get('n_numbers',20)
        # index for repeated benchmarks:
        self.exam_idx   = kwargs.get('exam_idx', 1)
        # number of problems in the benchmark
        self.n_problems = kwargs.get('n_problems', 360)
        # path to benchmark reports:
        self.results_path = self.path  + 'results/'
        # save blank benchmarks for repeated use:
        self.save_path = self.path  + 'saved/'
        self.figures_path = self.path  + 'figures/'

        create_missing_directory(self.path)
        create_missing_directory(self.save_path)
        create_missing_directory(self.results_path)
        create_missing_directory(self.figures_path)

        if '_' in self.exam_name:
            self.ci_method = (self.exam_name).split('_')[1]
            self.exam_name_wo_ci_method = (self.exam_name).split('_')[0]
        else:
            self.exam_name_wo_ci_method = self.exam_name

        if self.exam_name_wo_ci_method == 'SignificantFigures':

            self.problems = SignificantFigures(n_problems=self.n_problems)

        elif self.exam_name_wo_ci_method == 'StandardDeviation':

            self.problems = StandardDeviation(
                n_numbers=self.n_numbers,
                n_problems=self.n_problems
            )

        elif self.exam_name_wo_ci_method in ('MediatedCausalitySmoking',
                                             'MediatedCausality'):

            plot_path = self.save_path + exam_name + '_figures/'
            self.problems = MediatedCausality(
                plot_path,
                exam_name,
                plot_flag=True,
                n_problems=self.n_problems
            )

        # For grading and saving:
        # self.n_samples = self.problems.get_n_samples()
        # self.tables = self.problems.get_tables()
        # self.p_diff = self.problems.get_p_diff()
        # self.p_diff_ci_upper = self.problems.get_p_diff_ci_upper()
        # self.p_diff_ci_lower = self.problems.get_p_diff_ci_lower()
        # self.difficulty = self.problems.get_difficulty()
        # self.questions = self.problems.get_questions()
        # self.solutions = self.problems.get_solutions()
        # length_str = f"\n Number of questions: {self.n_problems}"
        # exam_str = '\n Exam: ' + exam_name
        # model_str = '\n Model: '
        # temp_str = '\n Temperature: ' + str(self.temperature)
        # effort_str = '\n Reasoning effort: ' + self.reasoning_effort
        #self.benchmark = RecordBenchmark(self.path,'none',self.problems) #<---- FIX THIS NEXT

        report = {
            "exam_name": exam_name,
            # "exam_str": exam_str,
            # "length_str": length_str,
            # "temp_str": temp_str,  
            # "model_str": model_str,
            "difficulty": self.problems.get_difficulty(),
            "questions": self.problems.get_questions(),
            "solutions": self.problems.get_solutions(),
            "exam_idx": self.exam_idx,
            "reuse": self.save_path, # what is this
            "checkpoints": self.checkpoints, # what is this
            "n_samples": self.problems.get_n_samples(),
            "tables": self.problems.get_tables(),
            "p_diff": self.problems.get_p_diff(),
            "p_diff_ci_upper": self.problems.get_p_diff_ci_upper(),
            "p_diff_ci_lower": self.problems.get_p_diff_ci_lower(),
            "difficulty": self.problems.get_difficulty()
        }

        # # save an npz file of this exam:
        # self.benchmark.save_blank_exam_npz(report)  #<---- FIX THIS NEXT, NEEDS TO SAVE WHATEVER IS IN report
        # if self.record_txt: # not recommended (files too big)
        #     self.benchmark.save_blank_exam_txt(report)
