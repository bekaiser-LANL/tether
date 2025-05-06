# Uncertainty Quantification by LLMs

While we refer to each examination as a benchmark, the goal is to characterize behavior (UQ approach, judgement) rather than to compare LLM results to ground truth.

# How to use this code

## Set your path for benchmark data
Prior to generating, running, or analyzing benchmarks (or running pytests) you need to set the path to your benchmarks directory. Do this in terminal with:

`export PATH_TO_BENCHMARKS=/$YOUR_PATH_TO_BENCHMARKS$/benchmark_results`

Alternatively, you can add the export command above into your ~/.bashrc or ~/.zshrc file and source them. In bash this looks like:

`echo 'export PATH_TO_BENCHMARKS="/$YOUR_PATH_TO_BENCHMARKS$/benchmark_results"' >> ~/.bashrc`

`source ~/.bashrc`

and you can check if it works with:

`echo $PATH_TO_BENCHMARKS`

## Generate a UQ benchmark

`python3 generate.py BENCHMARK`

For example:

`python3 generate.py MediatedCausality_tdist --n_problems=9 --make_plots`

will generate a saved benchmark MediatedCausality_tdist_0.npz in /PATH_TO_BENCHMARKS/benchmarks/saved/ and a figure for each problem in /PATH_TO_BENCHMARKS/benchmarks/saved/MediatedCausality_tdist_figures/.

## Run a benchmark

`python3 run.py BENCHMARK MODEL`

Be sure to include the index of the benchmark in BENCHMARK. For example, MediatedCausality_tdist_0 for the first MediatedCausality_tdist benchmark (you can repeated generate more of the same benchmark).

## Analyze a benchmark

The generic command for analyzing a benchmark is:

`python3 analyze.py BENCHMARK_NPZ_FILENAME_WITHOUT_SUFFIX`

Calling the command above just loads the completed benchmark .npz file, nothing more. In practice you will want to do analyses called with command line arguments. For example:

`python3 analyze.py MediatedCausality_tdist_0_mistral_0 --grade_estimate --verbose`

This will estimate the grade of the completed benchmark saved in `MediatedCausality_tdist_0_mistral_0.npz` and will output to terminal the grades as it loops over all questions in the benchmark. Some helpful command line arguments are:

`--print_vars`

...prints the variables saved in the .npz file to the terminal.

`--print_responses`

...prints every question, LLM response, and solution for the entire benchmark to terminal.

`--grade_estimate`

...uses a LLM grader and deterministic pattern-recognition grader to analyze LLM responses. If the LLM grader and the deterministic grader agree that the LLM response is correct/incorrect, then that correct/incorrect value is stored. In this case, a graded benchmark file is saved with the suffix `_final_grade.npz`. If the LLM grader and the deterministic grader disagree that the LLM response is correct/incorrect, then the problem is flagged for human review and the graded benchmark file is saved with the suffix `_provisional_grade.npz`.

`--human_review`

...loops over all problems that were flagged for human review with `--grade_estimate` and saved with the suffix `_provisional_grade.npz`. You can then manually label problems as correct/incorrect through command line interaction until all problems flagged for human review are done. Then the graded benchmark file is saved with the suffix `_final_grade.npz`.

# Models

## Adding your own model

## OpenAI models

o1 gpt-4o o3-mini gpt-4.5-preview

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

## StandardDeviation
Since the standard deviation appears in many confidence interval estimates, we provide a benchmark for it. Randomly generates sets of 10 integers selected from within -100 to 100 by default, then asks for the standard deviation. Population vs. sample standard deviation is not indicated, nor is the necessary precision. An example prompt:

"What is the standard deviation of -72 -25 -26 -51 -77 -67 15 -28 23 5?"

## MediatedCausality 

### No UQ method, no ontology 
No uncertainty quantification method specified in the prompt and no ontology specified in the prompt. There are two of these benchmarks, one for confidence intervals estimated from student's t distribution applied to the final probability difference (MediatedCausality_tdist) and one for confidence intervals estimated from bootstrapping the final probability difference (MediatedCausality_bootstrap).
An example prompt:

"Consider the following causal inference problem. The number of samples that do not X, do not Y, and do not Z is 1. 2 samples do not X, do not Y, and do Z. 1 samples do not X, do Y, and do not Z. 16 samples do not X, do Y, and do Z. 17 samples do X, do not Y, and do not Z. 1 samples do X, do not Y, and do Z. 1 samples do X, do Y, and do not Z. 1 samples do X, do Y, and do Z. Does doing X cause Y? Please answer 'A' for yes, 'B' for no, or 'C' for uncertain. Please use only the data provided here and the 95% confidence level."

### No UQ method in prompt, with ontology (does smoking cause lung cancer)

#### MediatedCausalitySmoking_bootstrap
#### MediatedCausalitySmoking_tdist
Example prompt:  
  
"Consider the following causal inference problem. The number of samples that do not smoke, do not have lung cancer, and do not have tar deposits in lungs is 9. 8 samples do not smoke, do not have lung cancer, and do have tar deposits in lungs. 34 samples do not smoke, do have lung cancer, and do not have tar deposits in lungs. 262 samples do not smoke, do have lung cancer, and do have tar deposits in lungs. 8 samples do smoke, do not have lung cancer, and do not have tar deposits in lungs. 31 samples do smoke, do not have lung cancer, and do have tar deposits in lungs. 3 samples do smoke, do have lung cancer, and do not have tar deposits in lungs. 240 samples do smoke, do have lung cancer, and do have tar deposits in lungs. Does smoking cause lung cancer? Answer 'A' for yes, 'B' for no, or 'C' for uncertain. Use only the data provided here and the 95% confidence level."

### UQ method in prompt, no ontology (does X cause Y)
#### MediatedCausalityWithMethod_bootstrap
#### MediatedCausalityWithMethod_tdist

# Testing

Running the complete suite of unit tests (36) with pytest takes about 5 minutes and 40 seconds.

# Licenses
This project is licensed under the [MIT License](LICENSE.md).

# Copyright
LANL O4894

© 2025. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
