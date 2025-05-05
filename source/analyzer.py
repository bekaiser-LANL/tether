""" Tools for analyzing saved benchmarks """
import os
import numpy as np
from source.utils import get_model_and_indices
from source.utils import detect_duplicate_tables
#import numpy as np
#import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import re
# import pandas as pd

data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

class Analyzer():
    """ Tools for analyzing saved benchmarks """

    def __init__(self, npz_filename, **kwargs):
        """ The benchmark name is the full name of the .npz file 
        without the suffix """
 
        self.exam_name = get_model_and_indices(npz_filename)[0]
        self.ci_method = get_model_and_indices(npz_filename)[1]
        self.exam_idx  = get_model_and_indices(npz_filename)[2]       
        self.model     = get_model_and_indices(npz_filename)[3]
        self.run_idx   = get_model_and_indices(npz_filename)[4]
        self.verbose = kwargs.get('verbose', False)
        self.print_vars = kwargs.get('print_vars', False)
        self.print_responses = kwargs.get('print_responses', False)
        self.completed_path = os.path.join(data_path, 'completed',self.model)
        self.npz_filepath = os.path.join(
            self.completed_path,
            npz_filename + '.npz'
        )

        # Load the .npz file
        self.data = np.load(self.npz_filepath, allow_pickle=True)

        if self.print_vars:
            self.print_keys()
        
        if self.print_responses:
            self.print_completed_benchmark()

        # if grade_with_openai
        # 1) pass through pattern recognizer
        # 2) Any result that fits a pattern but disagrees with correct is incorrect
        #    Otherwise, it's assumed correct.
        # 3) Ask openai for if the correct/incorrect label for a response is correct
        # 4) Flag any questions that don't match 
        # 5) Save first pass
        
        # 5) A separate script opens and runs the questions that don't match for interactive human review
        # 6) Save the final grade

    def print_keys(self):
            """ List all keys stored in the file """
            print("\n Keys:\n", self.data.files)
            #print("\n response:\n ", self.data["responses"])

    def print_completed_benchmark(self):
        """ Print the completed benchmark Q&A """
        n_problems = len(self.data["question"])
        for i in range(0,2):
            print('\n\n******************************************************')
            print('\n question = ',self.data["question"][i])
            print(' responses = ',self.data["responses"][i])
            print(' solution = ',self.data["solution"][i])
            print('\n')
            # print(' unbiased solution = ',data["unbiased_solution"][i])
            # print(' biased solution = ',data["biased_solution"][i])

    def verify_no_duplicates(self):
        if self.exam_name.startswith('MediatedCausality'):
            has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(self.data['table'])
            print(f"\n Benchmark: {self.exam_name}"
                f"\n Duplicate tables detected: {has_duplicates}"
                f"\n Number of problems: {n_problems}")
            if has_duplicates:
                print(f" {duplicate_pairs} duplicate pairs found")   
        print(f"\n Verify no duplicate problems needs to be implemented for {self.exam_name}")

    def grade_with_openai(self):
        # TO ADD: GRADE WITH OPENAI: 

        from openai import OpenAI # pylint: disable=import-outside-toplevel
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)

        # Verify a completed benchmarks:
        exam_idx = 0
        # exam_names = ['MediatedCausality_bootstrap_0_gpt-4.1',
        #               'MediatedCausality_tdist_0_gpt-4.1']
        # exam_names = ['MediatedCausality_bootstrap_0_o3',
        #               'MediatedCausality_tdist_0_o3']
        # exam_names = ['MediatedCausality_tdist_0_mistral']
        # exam_names = ['MediatedCausalityWithMethod_bootstrap_0_gpt-4.1',
        #               'MediatedCausalityWithMethod_tdist_0_gpt-4.1']    
        exam_names = ['MediatedCausalityWithMethod_bootstrap_0_gpt-4.1']    
        for i in range(0,len(exam_names)):
            data = load_saved_benchmark(data_path + '/completed/',exam_names[i], exam_idx)
            n_problems = len(data["question"])
            print('\n\n',exam_names[i])
            for j in range(0,3):
                # print('\n question =',data["question"][j])
                # print(' responses =',data["responses"][j])
                # print(' solution =',data["solution"][j])

                prompt = 'The correct answer is ' + data["solution"][j] + ', is the following response correct: ' + data["responses"][j] + '? Please just answer True or False'
                print('*********************************************************')
                print('\n prompt: ',prompt)
                response = ask_openai(prompt, client,'gpt-4o')
                print('\n gpt-4o grader: ',response)
                print('\n Correct answer: ',data["solution"][j])