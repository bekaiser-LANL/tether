# Tether: A suite of LLM benchmarks for ASC-related tasks 

To benchmark a model you need to choose the model (e.g., llama3.2), the interface (e.g., Ollama), the benchmarks you would like to run (e.g., 'standardDeviation') in run.py. Then you must choose generate = True or False and save = True or False for each benchmark. If generate = True then a new randomly-generated instantiation of the benchmark will be run. If generate = False, then a previously saved version of the benchmark will run using the corresponding text file in /benchmarks/saved/. If save = True the benchmark will be saved to /benchmarks/saved/.

# How to use Tether

To do

# Available benchmarks 

## standardDeviation 
Randomly generates sets of 20 integers, then asks the LLM to compute the standard deviation to 4 decimal places by default.

## significantFigures
Randomly generates floating point numbers and a random number of significant figures, then asks the LLM to express the number in scientific E notation using the correct number of significant figures. 

## mediatedCausalitySmoking 
Randomly generates tables of 3 binary variables corresponding to cause (X, smoking), effect (Y, lung cancer), and mediator (Z, tar in lungs) with randomly chosen sample sizes. Asks the LLM to determine if A) X causes Y with 95% confidence, B) X does not cause Y with 95% confidence, or C) the causal relationship is uncertain at the 95% confidence level. The correct answer is computed using the front door criterion and the standard error of proportion is used to compute the 95% confidence intervals.

## mediatedCausalitySmokingWithMethod
The same as mediatedCausalitySmoking but the prompt includes instructions on how to compute the correct answer: ``Does smoking cause lung cancer? Use the front-door criterion to determine if smoking causes cancer from the provided data. Use the standard error of proportion and full range error propagation to calculate the most conservative estimate of the 95% confidence level for the front-door criterion calculation. Use the 95% confidence level intervals to answer 'A' for yes, 'B' for no, or 'C' for uncertain.''

# Using Tether to benchmark LLaMa models with Ollama

1) Download ollama: https://ollama.com/download  
2) Open the downloaded zip, install   
3) Enter the following terminal command for each model you plan on using, for example:  

`$ ollama pull llama3`   

4) Now enter 

`$ ollama serve` 

and let it hang. This runs a server process that listens for API requests.

5) Now open another terminal, change the model choice in Tether/run.py to the llama model you pulled, and run.

# Licenses
This project is licensed under the [MIT License](LICENSE.md).

# Copyright
LANL O4894

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
