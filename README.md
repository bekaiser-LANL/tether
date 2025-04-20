# Tether: A suite of LLM benchmarks for scientific trustworthiness 

# How to use Tether

## Generate a benchmark

`python3 generate.py BENCHMARK /PATH/benchmarks/`

For example:

`python3 generate.py MediatedCausality_tdist /MY_PATH/Desktop/benchmarks/ --n_problems=9 --make_plots`

will generate a saved benchmark MediatedCausality_tdist_0.npz in /MY_PATH/Desktop/benchmarks/saved/ and a figure for each problem in /MY_PATH/Desktop/benchmarks/saved/MediatedCausality_tdist_figures/.

## Run a benchmark

`python3 run.py BENCHMARK MODEL /PATH/benchmarks/`

## Analyze a benchmark

# Models

## Adding your own model

## OpenAI models

'o1' 'gpt-4o' 'o3-mini' 'gpt-4.5-preview'

## Ollama models

 available.

1) Download ollama: https://ollama.com/download  
2) Open the downloaded zip, install   
3) Enter the following terminal command for each model you plan on using, for example:  

`$ ollama pull llama3`   

4) Now enter 

`$ ollama serve` 

and let it hang. This runs a server process that listens for API requests.

5) Now check if Tether has your ollama model and, if not, add the model to ollama_model_list in proctor.py.

6) Use your downloaded ollama model (e.g., 'llama3.2' 'llama3') to run.

# Benchmarks 

## SignificantFigures
Randomly generates floating point numbers and a random number of significant figures, then asks the LLM to express the number in scientific E notation using the correct number of significant figures. 

## 'StandardDeviation'
Randomly generates sets of 20 integers, then asks the LLM to compute the standard deviation to 4 decimal places by default.

## MediatedCausality 

### No UQ method in prompt, no ontology (does X cause Y)
#### MediatedCausality_tdist
#### MediatedCausality_bootstrap

### No UQ method in prompt, with ontology (does smoking cause lung cancer)

#### MediatedCausalitySmoking_bootstrap
#### MediatedCausalitySmoking_tdist
An example prompt:  
  
"Consider the following causal inference problem. The number of samples that do not smoke, do not have lung cancer, and do not have tar deposits in lungs is 9. 8 samples do not smoke, do not have lung cancer, and do have tar deposits in lungs. 34 samples do not smoke, do have lung cancer, and do not have tar deposits in lungs. 262 samples do not smoke, do have lung cancer, and do have tar deposits in lungs. 8 samples do smoke, do not have lung cancer, and do not have tar deposits in lungs. 31 samples do smoke, do not have lung cancer, and do have tar deposits in lungs. 3 samples do smoke, do have lung cancer, and do not have tar deposits in lungs. 240 samples do smoke, do have lung cancer, and do have tar deposits in lungs. Does smoking cause lung cancer? Please answer 'A' for yes, 'B' for no, or 'C' for uncertain. Please use only the data provided here and the 95% confidence level."

### UQ method in prompt, no ontology (does X cause Y)
#### MediatedCausalityWithMethod_bootstrap
#### MediatedCausalityWithMethod_tdist



# Licenses
This project is licensed under the [MIT License](LICENSE.md).

# Copyright
LANL O4894

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
