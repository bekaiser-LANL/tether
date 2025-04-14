import numpy as np
import math as ma
import random
import re
import os

class readSavedBenchmarkNpz():
    def __init__(self, read_path  ):
        self.read_path = read_path
  
        data = np.load(read_path,allow_pickle=True)
        self.exam_name = data['exam_name']
        self.questions = data['questions']
        self.solutions = data['solutions']
        self.model_str = data['model_str']
        self.exam_str = data['exam_str']
        self.n_problems = data['n_problems']

        if self.exam_name == 'significantFigures' or self.exam_name == 'standardDeviation':
            self.metadata = {
                "Name": self.exam_name,
                "n_problems": self.n_problems
            }
        elif self.exam_name == 'mediatedCausalitySmoking' or self.exam_name == 'mediatedCausalitySmokingWithMethod':
            self.metadata = {
                "Name": self.exam_name,
                "dP": data['dP'],
                "P_Y1doX1": data['P_Y1doX1'],
                "P_Y1doX0": data['P_Y1doX0'],
                "P_Y1doX1_CI": data['P_Y1doX1_CI'],
                "P_Y1doX0_CI": data['P_Y1doX0_CI'],
                "A_count": data['C_count'],
                "B_count": data['C_count'],
                "C_count": data['C_count'],
                "n_problems": self.n_problems
            }

    def get_questions(self): # all tests need this
        return self.questions

    def get_solutions(self): # all tests need this
        return self.solutions

    def get_metadata(self): # all tests need this
        return self.metadata

