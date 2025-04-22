from .grader import Grader
from .utils import load_saved_benchmark, get_npz_filename
from .utils import create_missing_directory
from .utils import strip_after_second_underscore
from .utils import get_after_second_underscore
import numpy as np
import subprocess
import requests
import os
from torchvision import transforms
from PIL import Image
from transformers import AutoConfig, MllamaProcessor, AutoTokenizer, AutoModelForVision2Seq, AutoModelForCausalLM, AutoProcessor, MllamaForConditionalGeneration

# remove generation from this

class Proctor():

    def __init__(self, exam_name, model, benchmark_path, **kwargs):

        self.benchmark_path = benchmark_path
        self.saved_benchmark_path = os.path.join(self.benchmark_path, 'saved')
        if exam_name.count("_") == 2: # includes exam_idx at end
            self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            self.exam_name = exam_name
            self.exam_idx = kwargs.get('exam_idx','unset')
        self.exam_name = exam_name
        self.model = model
        self.verbose = kwargs.get('verbose',False)
        self.modelpath = kwargs.get('model_path')
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
        #print(benchmark)
        self.questions = benchmark['question']
        #print("Value:", len(questions))
        #npzfile = np.load(benchmark, allow_pickle=True)
        #print("Keys in npz file:", npzfile.files)

        # To see the raw content
        #for key in npzfile.files:
        #    print(f"{key}: {type(npzfile[key])}") 
        #print("Type of benchmark:", type(benchmark))
        #print("Type of benchmark['question']:", type(benchmark['question']))
        responses = self.give_benchmark(benchmark)

        np.savez(self.npz_filename, **benchmark, responses=responses)

        self.record_txt = kwargs.get('record_txt', False) # save blank benchmark as .txt  
      
        #self.path       = settings['path'] + '/benchmarks/completed/' # path to benchmark reports
        #self.reuse      = settings['path'] + '/benchmarks/saved/' # path to saved benchmark
        #self.figures    = settings['path'] + '/benchmarks/figures/'
        #self.modelpath = kwargs.get('model_path')
        #self.grader     = grader()
        #self.generate   = settings['generate']
        #self.exam_idx   = settings['exam_idx']
            
        #self.create_missing_directory(self.path)
        #self.create_missing_directory(self.reuse)
        #self.create_missing_directory(self.figures)
        ollama_model_list = []
        openai_classic_model_list = [] 

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
        tmp = None
        model_path = os.path.join(self.modelpath, self.model)
        is_local_model = self.model in os.listdir(self.modelpath) and os.path.isdir(model_path)

        if not is_local_model:
            tmp = self.load_llm()

        responses = []

        if is_local_model:
            # Only call once for all questions
            responses = self.give_question_to_llm(None, tmp)  # local model handles its own loop
        else:
            # API models, call per-question
            for i, prompt in enumerate(self.questions):
                response = self.give_question_to_llm(prompt, tmp)
                if self.verbose:
                    print(f"\n Question {i}")
                    print(prompt)
                    print(response)
                responses.append(response)
        responses = np.array(responses)
        return responses

    def load_llm(self):
        """ Load & set up the LLM for benchmarking """
        tmp = None
        if self.model in ollama_model_list:
            with subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            ) as process:
                process.communicate()
            tmp = "http://localhost:11434/api/generate"
        elif self.model in openai_all_model_list:
            from openai import OpenAI # pylint: disable=import-outside-toplevel
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
        return tmp

    def give_question_to_llm(self, prompt, tmp):
        """ Method for prompting & recording LLMs """
        response = None
        local_models = os.listdir(self.modelpath)
        ollama_model_list = []
        openai_all_model_list = [] 
        if self.model in ollama_model_list:
            print("Model selected:", self.model)
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            # Send the request to the API
            request = requests.post(tmp, json=payload, timeout=120)
            if request.status_code == 200:
                # This is the standard HTTP status code for a successful request.
                # Successful response from the Ollama API
                response = request.json()["response"]
            else:
                print("Error:", request.status_code, request.text)
            return response
        elif self.model in openai_all_model_list:
            print("Model selected:", self.model)
            response = self.ask_openai(prompt,self.model)
            return response
        # locally downloaded LLMs
        elif self.model in local_models and os.path.isdir(os.path.join(self.modelpath, self.model)):
            print("Model selected:", self.model)
            responses = []
            # Load the model and tokenizer
            model = AutoModelForCausalLM.from_pretrained(self.modelpath + self.model)
            tokenizer = AutoTokenizer.from_pretrained(self.modelpath + self.model)
            # Set pad_token_id to eos_token_id if not already set
            if model.config.pad_token_id is None:
                model.config.pad_token_id = model.config.eos_token_id

            n = len(self.questions)
            for i in range(0,n): # length of exam

                prompt = self.questions[i]
                inputs = tokenizer(prompt, return_tensors="pt")

                # Generate response
                max_new_tokens = 5
                generate_ids = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens
                )
                response = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                if self.verbose:
                    print('\n Question ',i)
                    print(prompt)
                    print(response)
                responses.append(response)
            responses = np.array(responses)
            return responses
        #need to add multimodal models - these are only hardcoded for now; need to figure out better approach
        elif self.model == "Llama-3.2-90B-Vision-Instruct" or self.model == "Llama-3.2-11B-Vision":    
            # Load the model and tokenizer
            model = MllamaForConditionalGeneration.from_pretrained(self.modelpath + self.model)
            processor = AutoProcessor.from_pretrained(self.modelpath + self.model)
            model.tie_weights()
            self.path = self.imgpath
            # Get all image files in the directory (supports PNG, JPG, JPEG)
            image_files = [f for f in os.listdir(self.path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            responses = []
            # Iterate through images
            for index, img_file in enumerate(image_files):
                img_path = os.path.join(self.path, img_file)
                # Read image using OpenCV
                image = Image.open(img_path)
                print(img_file)

                # Convert back to PIL (if needed for processor)
                prompt = self.questions[index]#"<|begin_of_text|><|image|>Describe this equation"#                inputs = processor(images=image, text=prompt, return_tensors="pt")

               # Generate response from the model
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.9)

                # Decode the generated text
                response = processor.decode(outputs[0])
                responses.append(response)
            responses = np.array(responses)
            return responses
