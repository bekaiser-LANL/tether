""" Proctor administers benchmarks to LLMs """
import time
import subprocess
import requests
import gc # for efficient RAM use
import os
from .utils import load_saved_benchmark, get_npz_filename
from .utils import create_missing_directory
from .utils import get_after_second_underscore
import numpy as np
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM

ollama_model_list = ["llama3","llama3.2","mistral","granite3.2","deepseek-r1:32b","phi4","qwen3:30b","gemma3:27b-it-qat"]
openai_reasoning_model_list = ['o3-mini','o1','o3','o4-mini']
openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
openai_all_model_list = openai_reasoning_model_list + openai_classic_model_list
anthropic_model_list = ["claude-3-7-sonnet-20250219"]

def ensure_ollama_running():
    try:
        requests.get("http://localhost:11434")
    except requests.exceptions.ConnectionError:
        subprocess.Popen(["ollama", "serve"])
        time.sleep(5)

class Proctor():

    def __init__(self, exam_name, model, benchmark_path, **kwargs):

        self.benchmark_path = benchmark_path
        self.saved_benchmark_path = os.path.join(self.benchmark_path, 'blank')
        if exam_name.count("_") == 2: # includes exam_idx at end
            #self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            #self.exam_name = exam_name
            self.exam_idx = kwargs.get('exam_idx','unset')
        self.exam_name = exam_name
        self.model = model
        self.verbose = kwargs.get('verbose',False)
        self.modelpath = kwargs.get('model_path')
        self.results_path = os.path.join(self.benchmark_path, 'completed')
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
        self.n_numbers = kwargs.get('n_numbers',100)
        # exam index for multiple versions of the same exam:
        self.exam_idx = kwargs.get('exam_idx','unset')
        if self.exam_idx =='unset':
            self.exam_idx = 0
        create_missing_directory(self.results_path)
        create_missing_directory(self.saved_benchmark_path)

        benchmark = load_saved_benchmark(
            self.saved_benchmark_path,
            self.exam_name,
            self.exam_idx
        )
        self.questions = benchmark['question']
        responses = self.give_benchmark(benchmark)

        np.savez(self.npz_filename, **benchmark, responses=responses)

        self.record_txt = kwargs.get('record_txt', False) # save blank benchmark as .txt  

    def ask_anthropic(self, question, model_choice):
        """ Method for prompting & recording Anthropic products """
        if model_choice in anthropic_model_list:
            try:
                message = self.client.messages.create(
                    model=model_choice,
                    system="You are a scientist.",
                    max_tokens=1024,
                    messages = [
                        {"role": "user", "content": question}
                    ],
                    temperature=self.temperature # 0.0 (deterministic) to 1.0 (random)
                )
                return message.content[0].text
            except Exception as e: # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")        

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
        local_models = os.listdir(self.modelpath)
        if self.model in openai_all_model_list:
            from openai import OpenAI # pylint: disable=import-outside-toplevel
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
        elif self.model in anthropic_model_list:
            from anthropic import Anthropic # pylint: disable=import-outside-toplevel
            self.client = Anthropic()
        elif self.model in local_models and os.path.isdir(os.path.join(self.modelpath, self.model)):
            print("\n Local model:", self.model)
            # Load the model and tokenizer
            self.model_instance = AutoModelForCausalLM.from_pretrained(self.modelpath + self.model)
            self.tokenizer_instance = AutoTokenizer.from_pretrained(self.modelpath + self.model)
            # Set pad_token_id to eos_token_id if not already set
            if self.model_instance.config.pad_token_id is None:
                self.model_instance.config.pad_token_id = self.model_instance.config.eos_token_id
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
        local_models = os.listdir(self.modelpath)
        response = None
        #ollama_model_list = []
        #openai_all_model_list = [] 
        if self.model in ollama_model_list:
            print("\n Ollama model:", self.model)
            # This will work — as long as you have 'ollama serve' running
            # in one terminal and the model is on the list.
            #ensure_ollama_running()
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            # Send the request to the API
            request = requests.post(url, json=payload, timeout=120)
            gc.collect() # for efficient RAM use
            if request.status_code == 200:
                # This is the standard HTTP status code for a successful request.
                # Successful response from the Ollama API
                response = request.json()["response"]
            else:
                print("Error:", request.status_code, request.text)
            return response
        elif self.model in openai_all_model_list:
            print("\n OpenAI model:", self.model)
            response = self.ask_openai(prompt,self.model)
            return response
            # try:
            #     request = requests.post(url, json=payload, timeout=60)
            #     if request.status_code == 200:
            #         return request.json()["response"]
            #     else:
            #         print("Error:", request.status_code, request.text)
            # except requests.exceptions.RequestException as e:
            #     print("Request failed:", e)
        elif self.model in anthropic_model_list:
            print("\n Anthropic model:", self.model)
            response = self.ask_anthropic(prompt,self.model)
            return response
        elif self.model in local_models and os.path.isdir(os.path.join(self.modelpath, self.model)):
            inputs = self.tokenizer_instance(prompt, return_tensors="pt")

            # Generate response
            max_length = 1000
            generate_ids = self.model_instance.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
                max_length=max_length,
            )
            response = self.tokenizer_instance.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            return response
        else:
            return '\n Model not available'

# openai model:
# response = self.ask_openai(prompt,self.model)
# return response
