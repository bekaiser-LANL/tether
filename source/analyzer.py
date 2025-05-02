""" Tools for analyzing saved benchmarks """
import os
import numpy as np
from .utils import get_npz_filename
from .utils import get_after_second_underscore
from source.utils import detect_duplicate_tables, load_saved_benchmark
#import numpy as np
#import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import re
# import pandas as pd

data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

class Analyzer():
    """ Tools for analyzing saved benchmarks """

    def __init__(self, model, exam_name, **kwargs):
 
        self.model = model
        self.verbose = kwargs.get('verbose', False)
        if self.verbose:
            print('True!')

        #self.benchmark_path = data_path
        self.completed_path = os.path.join(data_path, 'completed')
        if exam_name.count("_") == 2: # includes exam_idx at end
            #self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            #self.exam_name = exam_name
            self.exam_idx = kwargs.get('exam_idx','unset')
        self.exam_name = exam_name
        self.model = model
        self.verbose = kwargs.get('verbose',False)
        self.completed_path = os.path.join(data_path, 'results')
        self.npz_filename = get_npz_filename(
            self.completed_path,
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

    def grade_with_openai():
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