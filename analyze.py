""" Analyze benchmark results """
import os
import argparse
from source.analyzer import Analyzer
#from source.utils import detect_duplicate_tables, load_saved_benchmark
from source.utils import get_parser

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
    """ Analyze the benchmark """
    parser = get_parser(script="analyze")
    args = parser.parse_args()
    kwargs = vars(args)
    npz_filename = kwargs.pop("exam_name")
    verbose = kwargs.pop("verbose", False)
    grade = kwargs.pop("grade", False)

    Analyzer(npz_filename, verbose=verbose, grade=grade, **kwargs)

if __name__ == "__main__":
    main()
