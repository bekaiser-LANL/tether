""" Analyze benchmark results """
import os
import argparse
from source.analyzer import Analyzer
from source.utils import detect_duplicate_tables, load_saved_benchmark

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")


def ask_openai(question, client, model_choice):
    """ Method for prompting & recording OpenAI products """
    openai_reasoning_model_list = ['o3-mini','o1','o3']
    openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
    #openai_all_model_list = openai_reasoning_model_list + openai_classic_model_list
    if model_choice in openai_classic_model_list:
        try:
            response = client.chat.completions.create(
                model=model_choice,  # gpt-4.5-preview, gpt-4o
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=0.0 #self.temperature # 0.0 (deterministic) to 1.0 (random)
            )
            return response.choices[0].message.content
        except Exception as e: # pylint: disable=broad-exception-caught
            return f"Error: {e}"
    elif model_choice in openai_reasoning_model_list:
        try:
            response = client.chat.completions.create(model=model_choice,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            reasoning_effort='high' #reasoning_effort # Options: 'low', 'medium', 'high')
            )
            return response.choices[0].message.content.strip()
        except Exception as e: # pylint: disable=broad-exception-caught
            return f"Error: {e}"
    else:
        return print("\n Model choice not available ")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the completed benchmark for a specified model."
    )
    parser.add_argument(
        "benchmark_name",
        help="Name of the benchmark to run, including its index (e.g., MediatedCausality_tdist_0)"
    )
    parser.add_argument(
        "model_name",
        help="Name of the model to test (e.g., gpt-4o)"
    )
    parser.add_argument(
        "--path",
        default=data_path,
        help=f"Path to the benchmarks directory (default: from PATH_TO_BENCHMARKS or '{data_path}')"
    )
    # add options to do Analyzer.duplicate_check()
    # add option to do Analyzer.print_benchmark()
    # add option to do Analyzer.grade_with_openai()

    args = parser.parse_args()

    Analyzer(args.path, args.model_name, args.benchmark_name)
    print(f"\n Analyses of benchmark '{args.benchmark_name}' for model '{args.model_name}' completed at: {args.path}")


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


    # # Verify the blank standard deviation benchmark:
    # exam_idx = 0
    # exam_name = 'StandardDeviation'
    # data = load_saved_benchmark(data_path + '/blank/',exam_name, exam_idx)
    # n_problems = len(data["question"])
    # for i in range(0,2):
    #     print('\n question = ',data["question"][i])
    #     print(' unbiased solution = ',data["unbiased_solution"][i])
    #     print(' biased solution = ',data["biased_solution"][i])


    # from openai import OpenAI # pylint: disable=import-outside-toplevel
    # openai_api_key = os.getenv("OPENAI_API_KEY")
    # client = OpenAI(api_key=openai_api_key)
    # # Verify a completed benchmarks:
    # exam_idx = 0
    # # exam_names = ['MediatedCausality_bootstrap_0_gpt-4.1',
    # #               'MediatedCausality_tdist_0_gpt-4.1']
    # # exam_names = ['MediatedCausality_bootstrap_0_o3',
    # #               'MediatedCausality_tdist_0_o3']
    # # exam_names = ['MediatedCausality_tdist_0_mistral']
    # # exam_names = ['MediatedCausalityWithMethod_bootstrap_0_gpt-4.1',
    # #               'MediatedCausalityWithMethod_tdist_0_gpt-4.1']    
    # exam_names = ['MediatedCausalityWithMethod_bootstrap_0_gpt-4.1']    
    # for i in range(0,len(exam_names)):
    #     data = load_saved_benchmark(data_path + '/completed/',exam_names[i], exam_idx)
    #     n_problems = len(data["question"])
    #     print('\n\n',exam_names[i])
    #     for j in range(0,3):
    #         # print('\n question =',data["question"][j])
    #         # print(' responses =',data["responses"][j])
    #         # print(' solution =',data["solution"][j])

    #         prompt = 'The correct answer is ' + data["solution"][j] + ', is the following response correct: ' + data["responses"][j] + '? Please just answer True or False'
    #         print('*********************************************************')
    #         print('\n prompt: ',prompt)
    #         response = ask_openai(prompt, client,'gpt-4o')
    #         print('\n gpt-4o grader: ',response)
    #         print('\n Correct answer: ',data["solution"][j])

if __name__ == "__main__":
    main()
