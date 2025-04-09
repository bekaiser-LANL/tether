""" Tether """

# change this to remove generation: run benchmarks on models only

# To do: 
#   - measure the l1 error of significantFigures and standardDeviation solutions (in analysis)
#   - add saves along the way and ability to restart 

# Model choices for benchmarking:
# OpenAI APIs: 'o1' 'gpt-4o' 'o3-mini' 'gpt-4.5-preview' 
# Meta / Ollama APIs: 'llama3.2' 'llama3' 

# Benchmark choices:
# 'mediatedCausalityArithmetic'  
# 'mediatedCausality_bootstrap' 
# 'mediatedCausalitySmoking_bootstrap' 
# 'mediatedCausalityWithMethod_bootstrap' 
# 'mediatedCausalityWithMethod_tdist' 
# 'mediatedCausality_tdist' 
# 'mediatedCausalitySmoking_tdist' 
# 'standardDeviation'
# 'significantFigures'

import subprocess
import requests
import time
import numpy as np
import re
from source import *

benchmark = 'mediatedCausalitySmoking'
model = 'llama3'
generate = False # True = generate a new benchmark, False = load a saved benchmark .npz
n_problems = 180 # number of problems in the exam

settings = {
    "model": model,
    "generate": generate, 
    "exam_idx": 1, # leave 1 unless you need to make multiples of the same exam
    "path": '/Users/l281800/Desktop/', # path for the benchmark reports (i.e., results)  
    "n_problems": n_problems, # number of problems in the benchmark
}

proctor(settings, benchmark) #, checkpoints=2, restart=4) 

# #===============================================================================
# # standardDeviation study

# benchmark = 'standardDeviation'
# model = 'gpt-4o'
# generate = True # True = generate a new benchmark, False = load a saved benchmark .npz
# n_problems = 100 # number of problems in the exam
# n_numbers = 2 

# settings = {
#     "model": model,
#     "generate": generate, 
#     "exam_idx": 2, # leave 1 unless you need to make multiples of the same exam
#     "path": '/Users/l281800/Desktop/', # path for the benchmark reports (i.e., results)
#     "record_txt": False, # save blank benchmark as .txt        
#     "n_problems": n_problems, # number of problems in the exam
#     "temperature": 0.0, # OpenAI non-reasoning model temperature
#     "reasoning_effort": 'high', # OpenAI reasoning model effort 
#     "n_numbers": n_numbers # number of number for standardDeviation benchmark
# }

# for i in range(2,21):
#     settings["exam_idx"] = i 
#     settings["n_numbers"] = i
#     proctor(settings, benchmark)




print('\n Tether ran to completion! \n ')
