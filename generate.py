""" Randomly generate new benchmarks """

# Benchmark choices:
# 'MediatedCausalityArithmetic'  
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
benchmark_dir = '/Users/l281800/Desktop/benchmarks/'

Generator(benchmark_dir, exam_name, n_problems=9)  

print(f"\n {exam_name} benchmark generated! \n")