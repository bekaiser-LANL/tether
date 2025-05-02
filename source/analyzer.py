""" Tools for analyzing saved benchmarks """
import os
import numpy as np
from source.utils import get_model_and_indices
#from source.utils import load_saved_benchmark
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
        self.completed_path = os.path.join(data_path, 'completed',self.model)
        self.npz_filepath = os.path.join(
            self.completed_path,
            npz_filename + '.npz'
        )

        # Load the .npz file
        self.data = np.load(self.npz_filepath, allow_pickle=True)

        self.print_data_keys()

    def print_data_keys(self):
            """ List all keys stored in the file """
            print("\n Keys:\n", self.data.files)
            #print("\n response:\n ", self.data["responses"])

    def print_completed_benchmark(self):
        a=1
        # TO ADD: 
        # # Verify the blank standard deviation benchmark:
        # exam_idx = 0
        # exam_name = 'StandardDeviation'
        # data = load_saved_benchmark(data_path + '/blank/',exam_name, exam_idx)
        # n_problems = len(data["question"])
        # for i in range(0,2):
        #     print('\n question = ',data["question"][i])
        #     print(' unbiased solution = ',data["unbiased_solution"][i])
        #     print(' biased solution = ',data["biased_solution"][i])

    def verify_no_duplicates(self):
        a=1
        # TO ADD: 
        # # Verify that each mediated causality benchmark has no duplicate problems:
        # exam_idx = 0
        # exam_names = ['MediatedCausality_bootstrap',
        #               'MediatedCausalitySmoking_bootstrap',
        #               'MediatedCausalityWithMethod_bootstrap',
        #               'MediatedCausality_tdist',
        #               'MediatedCausalitySmoking_tdist', # <- has a duplicate
        #               'MediatedCausalityWithMethod_tdist'
        #               ]
        # for i in range(0,len(exam_names)):
        #     data = load_saved_benchmark(data_path + '/blank/',exam_names[i], exam_idx)
        #     has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(data['table'])
        #     print(f"\n Benchmark: {exam_names[i]}"
        #         f"\n Duplicate tables detected: {has_duplicates}"
        #         f"\n Number of problems: {n_problems}")
        #     if has_duplicates:
        #         print(f" {duplicate_pairs} duplicate pairs found")

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