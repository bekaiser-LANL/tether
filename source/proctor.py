""" Proctor administers benchmarks to LLMs """
import os
import subprocess
import requests
import time
import numpy as np
from .utils import load_saved_benchmark, get_npz_filename
from .utils import create_missing_directory
from .utils import strip_after_second_underscore
from .utils import get_after_second_underscore

ollama_model_list = ["llama3","llama3.2"]
openai_reasoning_model_list = ['o3-mini','o1','o3']
openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
openai_all_model_list = openai_reasoning_model_list + openai_classic_model_list

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        subprocess.Popen(["ollama", "serve"])
        time.sleep(5)

class Proctor():
    """ Administers benchmarks to LLMs """
    # pylint: disable=too-many-instance-attributes

    def __init__(self, benchmark_path, model, exam_name, **kwargs):

        self.benchmark_path = benchmark_path
        self.saved_benchmark_path = os.path.join(self.benchmark_path, 'saved')
        if exam_name.count("_") == 2: # includes exam_idx at end
            #self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            #self.exam_name = exam_name
            self.exam_idx = kwargs.get('exam_idx','unset')
        self.exam_name = exam_name
        self.model = model
        self.verbose = kwargs.get('verbose',False)
        self.results_path = os.path.join(self.benchmark_path, 'results')
        self.client = None
        self.npz_filename = get_npz_filename(
            self.results_path,
            self.exam_name,
            self.exam_idx,
            self.model
        )
        # Checkpoint frequency if an integer, no checkpoint .npz output if a NaN:
        self.checkpoint_freq = kwargs.get('checkpoint_freq','unset')
        # Restart question number if an integer, start at question 1 if a NaN:
        self.restart_idx = kwargs.get('restart_idx','unset')
        # for OpenAI reasoning models only:
        self.reasoning_effort = kwargs.get('reasoning_effort', 'high')
        # for OpenAI non-reasoning models only:
        self.temperature  = kwargs.get('temperature', 0.0)
        # number of numbers for standardDeviation benchmark:
        self.n_numbers = kwargs.get('n_numbers',20)
        # exam index for multiple versions of the same exam:
        self.exam_idx = kwargs.get('exam_idx','unset')

        create_missing_directory(self.results_path)
        create_missing_directory(self.saved_benchmark_path)

        benchmark = load_saved_benchmark(
            self.saved_benchmark_path,
            self.exam_name,
            self.exam_idx
        )

        responses = self.give_benchmark(benchmark)

        np.savez(self.npz_filename, **benchmark, responses=responses)

    def ask_openai(self, question, model_choice):
        """ Method for prompting & recording OpenAI products """
        if model_choice in openai_classic_model_list:
            try:
                response = self.client.chat.completions.create(
                    model=model_choice,  # gpt-4.5-preview, gpt-4o
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question}
                    ],
                    temperature=self.temperature # 0.0 (deterministic) to 1.0 (random)
                )
                return response.choices[0].message.content
            except Exception as e: # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        elif model_choice in openai_reasoning_model_list:
            try:
                response = self.client.chat.completions.create(model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                reasoning_effort=self.reasoning_effort # Options: 'low', 'medium', 'high')
                )
                return response.choices[0].message.content.strip()
            except Exception as e: # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")

    def give_benchmark(self, benchmark):
        """ Give all of the questions to the LLM """
        if self.model in openai_all_model_list:
            from openai import OpenAI # pylint: disable=import-outside-toplevel
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
        n = len(benchmark['question'])
        responses = []
        for i in range(0,n):
            prompt = benchmark["question"][i]
            response = self.give_question_to_llm(prompt)
            if self.verbose:
                print('\n Question ',i)
                print(prompt)
                print(response)
            responses.append(response)
        responses = np.array(responses)
        return responses

    def give_question_to_llm(self, prompt):
        """ Method for prompting & recording LLMs """
        response = None
        if self.model in ollama_model_list:
            print(" Ollama is broken ")
            # This will work — as long as you manually run `ollama run llama3'
            # in one terminal, and then execute this Python script from another.
            #ensure_ollama_running()
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            try:
                request = requests.post(url, json=payload, timeout=60)
                if request.status_code == 200:
                    return request.json()["response"]
                else:
                    print("Error:", request.status_code, request.text)
            except requests.exceptions.RequestException as e:
                print("Request failed:", e)

        # openai model:
        #if self.model in openai_all_model_list:
        response = self.ask_openai(prompt,self.model)
        return response
