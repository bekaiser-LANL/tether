""" Randomly generate new benchmarks """

# Needs to be set up so you can do: python3 generate.py 'model_name' with kwargs

# Benchmark choices: 
# 'MediatedCausality_bootstrap' 
# 'MediatedCausalitySmoking_bootstrap' 
# 'MediatedCausalityWithMethod_bootstrap' 
# 'MediatedCausalityWithMethod_tdist' 
# 'MediatedCausality_tdist' 
# 'MediatedCausalitySmoking_tdist' 
# 'StandardDeviation'
# 'SignificantFigures'

from source.generator import Generator

exam_name = 'MediatedCausalitySmoking_tdist'
path = '/Users/l281800/Desktop/benchmarks/'

Generator(path, exam_name, n_problems=9, plot_flag=True)  

print(f"\n {exam_name} benchmark generated! \n")