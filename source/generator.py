from .grader import grader
from .recorder import RecordBenchmark
from .utils import create_missing_directory
import numpy as np

class Generator():

    def __init__(self, settings, exam_name, **kwargs):

        self.exam_name = exam_name

        # default settings:
        self.checkpoints = kwargs.get('checkpoints', np.nan) # Checkpoint frequency if an integer, no checkpoint .npz output if a NaN
        self.restart = kwargs.get('restart', np.nan) # Restart question number if an integer, start at question 1 if a NaN
        self.reasoning_effort = kwargs.get('reasoning_effort', 'high') # for OpenAI reasoning models only
        self.temperature  = kwargs.get('temperature', 0.0) # for OpenAI non-reasoning models only
        self.record_txt = kwargs.get('record_txt', False) # save blank benchmark as .txt  
        self.n_numbers = kwargs.get('n_numbers',20) # number of numbers for standardDeviation benchmark
      
        self.path       = settings['path'] + '/benchmarks/completed/' # path to benchmark reports
        self.reuse      = settings['path'] + '/benchmarks/saved/' # path to saved benchmark
        self.figures    = settings['path'] + '/benchmarks/figures/'
        self.n_problems = settings['n_problems']
        self.exam_idx   = settings['exam_idx']
            
        create_missing_directory(self.path)
        create_missing_directory(self.reuse)
        create_missing_directory(self.figures)

        if '_' in self.exam_name:
            self.CI_method = (self.exam_name).split('_')[1]
            self.exam_name_wo_CI_method = (self.exam_name).split('_')[0]
        else:  
            self.exam_name_wo_CI_method = self.exam_name   

        if self.exam_name_wo_CI_method == 'significantFigures':

            from .benchmarks.significantFigures import significantFigures
            self.exam = significantFigures(n_problems=self.n_problems)

        elif self.exam_name_wo_CI_method == 'standardDeviation':

            from .benchmarks.standardDeviation import standardDeviation
            self.exam = standardDeviation(n_numbers=self.n_numbers, n_problems=self.n_problems)

        elif self.exam_name_wo_CI_method == 'mediatedCausalitySmoking' or self.exam_name_wo_CI_method == 'mediatedCausality':

            from .benchmarks.mediated_causality import MediatedCausality

            x_name = 'smoke'
            z_name = 'have tar deposits in lungs'
            y_name = 'have lung cancer'
            x_name_verb = 'smoking'
            y_name_noun = 'lung cancer'
            name_list = [x_name,z_name,y_name,x_name_verb,y_name_noun]
            plot_path = self.reuse + exam_name + '_figures/'
            self.exam = MediatedCausality(plot_path,exam_name,name_list=name_list,plot_flag=True,n_problems=self.n_problems)


        # For grading and saving:
        self.metadata  = self.exam.get_metadata()
        self.questions = self.exam.get_questions()
        self.solutions = self.exam.get_solutions()
        length_str = '\n Number of questions: %i' %self.n_problems
        exam_str = '\n Exam: ' + exam_name
        model_str = '\n Model: '     
        temp_str = '\n Temperature: ' + str(self.temperature)
        effort_str = '\n Reasoning effort: ' + self.reasoning_effort
        self.benchmark = RecordBenchmark(self.path,'none',self.exam)

        report = {
            "exam_name": exam_name,
            "exam_str": exam_str,
            "length_str": length_str,
            "temp_str": temp_str,  
            "model_str": model_str,  
            "effort_str": effort_str,                       
            "questions": self.questions,
            "solutions": self.solutions,      
            "exam_idx": self.exam_idx,
            "reuse": self.reuse,
            "checkpoints": self.checkpoints,
            "metadata": self.metadata          
        }

        # save an npz file of this exam:
        self.benchmark.save_blank_exam_npz(report) 
        if self.record_txt: # not recommended
            self.benchmark.save_blank_exam_txt(report)            
