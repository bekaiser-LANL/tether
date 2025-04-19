""" Run a previously generated benchmark on an LLM """

# * Model choices:
# OpenAI APIs: 
#   'o1' 'gpt-4o' 'o3-mini' 'gpt-4.5-preview' 
# Meta / Ollama APIs: 
#   'llama3.2' 'llama3' 

# * Benchmark choices:
# 'MediatedCausalityArithmetic'
# 'MediatedCausality_bootstrap'
# 'MediatedCausalitySmoking_bootstrap'
# 'MediatedCausalityWithMethod_bootstrap' 
# 'MediatedCausalityWithMethod_tdist'
# 'MediatedCausality_tdist'
# 'MediatedCausalitySmoking_tdist'
# 'StandardDeviation'
# 'SignificantFigures'

from source.proctor import Proctor

benchmark = 'MediatedCausalitySmoking_tdist'
#model = 'llama3'
model = 'gpt-4o'

path_to_benchmarks = '/Users/l281800/Desktop/benchmarks/'
Proctor(path_to_benchmarks, model, benchmark, verbose=True)

print('\n Tether ran to completion! \n ')
