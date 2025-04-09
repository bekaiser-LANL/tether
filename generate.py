""" Tether """

# generate the benchmarks only

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

import numpy as np
from source import *

exam_name = 'mediatedCausalitySmoking'
generate = False # True = generate a new benchmark, False = load a saved benchmark .npz
n_problems = 180 # number of problems in the exam

settings = {
    "model": model,
    "generate": generate, 
    "exam_idx": 1, # leave 1 unless you need to make multiples of the same exam
    "path": '/Users/l281800/Desktop/', # path for the benchmark reports (i.e., results)  
    "n_problems": n_problems, # number of problems in the benchmark
}

proctor(settings, exam_name) #, checkpoints=2, restart=4) 


print('\n Tether ran to completion! \n ')
