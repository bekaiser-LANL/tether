""" Tools for analyzing saved benchmarks """
import os
import numpy as np
from .utils import get_npz_filename
from .utils import get_after_second_underscore
from source.grader import Grader
#import numpy as np
#import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import re
# import pandas as pd

# add bootstrapped confidence
# add l1 norm error for standard deviation

class Analyzer():
    """ Tools for analyzing saved benchmarks """

    def __init__(self, benchmark_path, model, exam_name, **kwargs):
 
        self.model = model
        self.benchmark_path = benchmark_path
        self.results_benchmark_path = os.path.join(self.benchmark_path, 'results')
        if exam_name.count("_") == 2: # includes exam_idx at end
            #self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            #self.exam_name = exam_name
            self.exam_idx = kwargs.get('exam_idx','unset')
        self.exam_name = exam_name
        self.model = model
        self.verbose = kwargs.get('verbose',False)
        self.results_path = os.path.join(self.benchmark_path, 'results')
        self.npz_filename = get_npz_filename(
            self.results_path,
            self.exam_name,
            self.exam_idx,
            self.model
        )

        # Load the .npz file
        data = np.load(self.npz_filename, allow_pickle=True)

        # List all keys stored in the file
        #if self.verbose:
        print("\n Keys:\n", data.files)

        print("\n response:\n ", data["responses"])

        # # Access individual arrays
        # print("\n\n First question:\n ", data["question"][0])
        # print("\n First response:\n ", data["responses"][0])
        # print("\n First solution:\n ", data["solution"][0])
        # print("\n\n Last question:\n ", data["question"][-1])
        # print("\n Last response:\n ", data["responses"][-1])
        # print("\n Last solution:\n ", data["solution"][-1])
        # print("\n Last p_diff:\n ", data["p_diff"][-1])     

        # if self.exam_name.startswith(("MediatedCausalitySmoking",
        #                               "MediatedCausalitySmokingWithMethod")):
        #     self.data = self.getMediatedCausalityResults()


    # def get_data(self):
    #     return self.data

    # def get_str_indices(self,lst,str):
    #     return [index for index, value in enumerate(lst) if value == str]

    # def get_grade(self,correct,llm):
    #     score = 0.
    #     count = 0.
    #     each = np.zeros([len(correct)])
    #     for i in range(len(correct)):
    #         count += 1.
    #         if correct[i] == llm[i]:
    #             score += 1.
    #             each[i] = 1.
    #     return score/count, each

    # def get_N_samples(self,questions):
    #     """ For MediatedCausality benchmarks """
    #     N_samples = np.zeros([len(questions)],dtype=int)
    #     for i in range(0,len(questions)):
    #         tmp = extract_numbers(questions[i])
    #         N_samples[i] = np.sum(tmp[0:8],dtype=int)
    #     return N_samples

    # def extract_numbers(self,s):
    #     return [int(num) for num in re.findall(r'\d+', s)]

    # def getMediatedCausalityResults(self):

    #     read_path = self.path + self.model + '/' + self.benchmark + '_' + str(self.exam_idx) + '.npz'
    #     data = np.load(read_path,allow_pickle=True)
    #     self.exam_name = data['exam_name']
    #     self.questions = data['questions']
    #     self.solutions = data['solutions']
    #     self.response  = data['response'] 
    #     self.correct   = data['correct']               
    #     self.model_str = data['model_str']
    #     self.exam_str = data['exam_str']
    #     self.n_problems = data['n_problems']
    #     self.P_Y1doX1 = data['P_Y1doX1']
    #     self.P_Y1doX1_CI = data['P_Y1doX1_CI']
    #     self.P_Y1doX0 = data['P_Y1doX0']
    #     self.P_Y1doX0_CI = data['P_Y1doX0_CI']

    #     correct = 0
    #     A_correct = 0; A_problem = 0;
    #     B_correct = 0; B_problem = 0;
    #     C_correct = 0; C_problem = 0;
        
    #     for iii in range(0,len(self.response)):
    #         if self.solutions[iii] == 'A':
    #             A_problem += 1
    #             if self.response[iii] == self.solutions[iii]:
    #                 A_correct += 1; correct += 1
    #         elif self.solutions[iii] == 'B':
    #             B_problem += 1
    #             if self.response[iii] == self.solutions[iii]:
    #                 B_correct += 1; correct += 1
    #         elif self.solutions[iii] == 'C':
    #             C_problem += 1
    #             if self.response[iii] == self.solutions[iii]:
    #                 C_correct += 1; correct += 1
    #     #print(' A_problem+B_problem+C_problem-len(solution) = ',A_problem+B_problem+C_problem-len(solution))
    #     # print(A_correct,A_problem)
    #     # print(B_correct,B_problem)
    #     # print(C_correct,C_problem)
    #     A_score = A_correct / A_problem * 100.
    #     B_score = B_correct / B_problem * 100.
    #     C_score = C_correct / C_problem * 100.
    #     self.total_score = correct / len(self.solutions) * 100.

    #     print(' total_score = ',self.total_score)

    #     self.A_correct = A_correct # number of A problems answered correctly
    #     self.A_problem = A_problem # count of problems with A as the correct answer 
    #     self.A_score   = A_score # percentage of A problems answered correctly

    #     self.B_correct = B_correct
    #     self.B_problem = B_problem 
    #     self.B_score   = B_score 

    #     self.C_correct = C_correct
    #     self.C_problem = C_problem 
    #     self.C_score   = C_score        

    #     # check the lengths of all of these variables
    #     data = {
    #         "solutions": self.solutions,
    #         "response": self.response,
    #         "questions": self.questions,
    #         "P_Y1doX1": self.P_Y1doX1,
    #         "P_Y1doX1_CI": self.P_Y1doX1_CI,
    #         "P_Y1doX0": self.P_Y1doX0,
    #         "P_Y1doX0_CI": self.P_Y1doX0_CI,
    #         "A_correct": self.A_correct,
    #         "A_problem": self.A_problem,
    #         "A_score": self.A_score,
    #         "B_correct": self.B_correct,
    #         "B_problem": self.B_problem,
    #         "B_score": self.B_score,
    #         "C_correct": self.C_correct,
    #         "C_problem": self.C_problem,
    #         "C_score": self.C_score,
    #         "total_score": self.total_score
    #     }
    #     return data
