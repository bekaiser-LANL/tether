""" Proctor administers benchmarks to LLMs """
import gc  # for efficient RAM use
import json
import os
import re
import subprocess
import time

import numpy as np
import requests
from langchain.agents import AgentType, Tool, initialize_agent
from langchain_core.exceptions import OutputParserException
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from tether.core.utils import (
    create_missing_directory,
    get_after_second_underscore,
    get_npz_filename,
    load_saved_benchmark,
)

# make sure the models you want to run are listed here
ollama_model_list = [
    "llama3:8b",
    "codellama:34b-python",
    "granite3.2",
    "deepseek-r1:32b",
    "phi4",
    "qwen2-math",
]
openai_reasoning_model_list = ["o3-mini", "o1", "o3", "o4-mini"]
openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
openai_all_model_list = openai_reasoning_model_list + openai_classic_model_list
anthropic_model_list = ["claude-3-7-sonnet-20250219"]


def ensure_ollama_running():
    """Check ollama is running"""
    try:
        requests.get("http://localhost:11434", timeout=60)
    except requests.exceptions.ConnectionError:
        subprocess.Popen(["ollama", "serve"])  # pylint: disable=consider-using-with
        time.sleep(5)


class Proctor:
    """Give the benchmark tests to the LLM"""

    def __init__(self, exam_name, model, benchmark_path, **kwargs):
        self.benchmark_path = benchmark_path
        self.saved_benchmark_path = os.path.join(self.benchmark_path, "blank")
        if exam_name.count("_") == 2:  # includes exam_idx at end
            # self.exam_name = strip_after_second_underscore(exam_name)
            self.exam_idx = int(get_after_second_underscore(exam_name))
        else:
            # self.exam_name = exam_name
            self.exam_idx = kwargs.get("exam_idx", "unset")
        self.exam_name = exam_name
        self.model = model
        self.agent = None
        if kwargs.get("agent", False):
            self.agent_flag = True
            if self.model in ollama_model_list:
                langchain_llm = ChatOllama(model=self.model)
                self.agent = initialize_agent(
                    tools=ToolRegistry.get_tools(),
                    llm=langchain_llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,
                    early_stopping_method="generate",
                    return_intermediate_steps=True,
                )
            elif self.model in openai_all_model_list:
                langchain_llm = ChatOpenAI(model=self.model)
                self.agent = initialize_agent(
                    tools=ToolRegistry.get_tools(),
                    llm=langchain_llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=True,
                    handle_parsing_errors=True,
                    max_iterations=10,
                    early_stopping_method="generate",
                    return_intermediate_steps=True,
                )
            else:
                raise ValueError(f"Unsupported model '{self.model}' for agent mode.")
        else:
            self.agent_flag = False
        self.agent_flag = kwargs.get("agent", False)
        self.verbose = kwargs.get("verbose", False)
        self.modelpath = kwargs.get("model_path")
        self.results_path = os.path.join(self.benchmark_path, "completed")
        self.client = None
        if self.agent:
            self.npz_filename = get_npz_filename(
                self.results_path, self.exam_name + "_agent", self.exam_idx, self.model
            )
        else:
            self.npz_filename = get_npz_filename(
                self.results_path, self.exam_name, self.exam_idx, self.model
            )
        print(self.model)
        # Checkpoint frequency if an integer, no checkpoint .npz output if a NaN:
        self.checkpoint_freq = kwargs.get("checkpoint_freq", "unset")
        # Restart question number if an integer, start at question 1 if a NaN:
        self.restart_idx = kwargs.get("restart_idx", "unset")
        # for OpenAI reasoning models only:
        self.reasoning_effort = kwargs.get("reasoning_effort", "high")
        # for OpenAI non-reasoning models only:
        self.temperature = kwargs.get("temperature", 0.0)
        # number of numbers for standardDeviation benchmark:
        self.n_numbers = kwargs.get("n_numbers", 100)
        # exam index for multiple versions of the same exam:
        self.exam_idx = kwargs.get("exam_idx", "unset")
        if self.exam_idx == "unset":
            self.exam_idx = 0
        create_missing_directory(self.results_path)
        create_missing_directory(self.saved_benchmark_path)

        benchmark = load_saved_benchmark(
            self.saved_benchmark_path, self.exam_name, self.exam_idx
        )
        self.questions = benchmark["question"]
        responses = self.give_benchmark(benchmark)
        np.savez(self.npz_filename, **benchmark, responses=responses)
        self.record_txt = kwargs.get(
            "record_txt", False
        )  # save blank benchmark as .txt

    def ask_anthropic(self, question, model_choice):
        """Method for prompting & recording Anthropic products"""
        if model_choice in anthropic_model_list:
            try:
                message = self.client.messages.create(
                    model=model_choice,
                    system="You are a scientist.",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": question}],
                    temperature=self.temperature,  # 0.0 (deterministic) to 1.0 (random)
                )
                return message.content[0].text
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")

    def ask_openai(self, question, model_choice):
        """Method for prompting & recording OpenAI products"""
        if model_choice in openai_classic_model_list:
            try:
                response = self.client.chat.completions.create(
                    model=model_choice,  # gpt-4.5-preview, gpt-4o
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    temperature=self.temperature,  # 0.0 (deterministic) to 1.0 (random)
                )
                return response.choices[0].message.content
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        elif model_choice in openai_reasoning_model_list:
            try:
                response = self.client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    reasoning_effort=self.reasoning_effort,  # Options: 'low', 'medium', 'high')
                )
                return response.choices[0].message.content.strip()
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")

    def give_benchmark(self, benchmark):
        """Give all of the questions to the LLM"""
        local_models = os.listdir(self.modelpath)
        if self.model in openai_all_model_list:
            from openai import OpenAI  # pylint: disable=import-outside-toplevel

            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.client = OpenAI(api_key=openai_api_key)
        elif self.model in anthropic_model_list:
            from anthropic import Anthropic  # pylint: disable=import-outside-toplevel

            self.client = Anthropic()
        elif self.model in local_models and os.path.isdir(
            os.path.join(self.modelpath, self.model)
        ):
            print("\n Local model:", self.model)
            # Load the model and tokenizer
            self.model_instance = AutoModelForCausalLM.from_pretrained(
                self.modelpath + self.model
            )
            self.tokenizer_instance = AutoTokenizer.from_pretrained(
                self.modelpath + self.model
            )
            # Set pad_token_id to eos_token_id if not already set
            if self.model_instance.config.pad_token_id is None:
                self.model_instance.config.pad_token_id = (
                    self.model_instance.config.eos_token_id
                )
        n = len(benchmark["question"])
        responses = []
        for i in range(0, n):
            prompt = benchmark["question"][i]
            if self.verbose:
                print("\n Question ", i)
                print(prompt)
            if self.agent:
                agent_prompt = self.build_agent_prompt(prompt)
                try:
                    raw_response = self.agent.invoke(agent_prompt)
                    response_text = (
                        raw_response["output"]
                        if isinstance(raw_response, dict)
                        else str(raw_response)
                    )
                except OutputParserException as e:
                    response_text = str(e)

                # Attempt to extract Final Answer from plain text (e.g., "Final Answer: B")
                fa_match = re.search(
                    r"Answer\s*[:\-]?\s*\(?([A-Ca-c])\)?", response_text, re.IGNORECASE
                )
                final_answer = fa_match.group(1).upper() if fa_match else None

                # Try to find a full JSON block with or without triple backticks
                json_match = re.search(
                    r'\{[^{}]*"answer"\s*:\s*"([A-Ca-c])"[^{}]*\}', response_text
                )
                parsed_json = None
                json_error = None
                json_answer = None
                json_explanation = None

                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(0))
                        json_answer = parsed_json.get("answer", "").upper()
                        json_explanation = parsed_json.get("explanation", None)
                    except json.JSONDecodeError as e:
                        json_error = f"JSON parse error: {e}"

                response_to_save = {
                    "output": response_text,
                    "final_answer": json_answer or final_answer,
                    "explanation": json_explanation,
                    "intermediate_steps": response_text.splitlines(),
                }

                if json_error:
                    response_to_save["json_error"] = json_error
                elif not json_match and not (json_answer or final_answer):
                    response_to_save["json_error"] = "No recognizable answer found"
                response = response_to_save
                print(response)
            else:
                response = self.give_question_to_llm(prompt)
                print(response)
            responses.append(response)
        responses = np.array(responses)
        return responses

    def give_question_to_llm(self, prompt):
        """Method for prompting & recording LLMs"""
        local_models = os.listdir(self.modelpath)
        response = None
        if self.model in ollama_model_list:
            print("\n Ollama model:", self.model)
            # This will work — as long as you have 'ollama serve' running
            # in one terminal and the model is on the list.
            # ensure_ollama_running()
            url = "http://localhost:11434/api/generate"
            payload = {"model": self.model, "prompt": prompt, "stream": False}
            # Send the request to the API
            request = requests.post(url, json=payload, timeout=240)
            gc.collect()  # for efficient RAM use
            if request.status_code == 200:
                # This is the standard HTTP status code for a successful request.
                # Successful response from the Ollama API
                response = request.json()["response"]
                return response
            else:
                print("Error:", request.status_code, request.text)
                return response
        elif self.model in openai_all_model_list:
            print("\n OpenAI model:", self.model)
            response = self.ask_openai(prompt, self.model)
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
            response = self.ask_anthropic(prompt, self.model)
            return response
        elif self.model in local_models and os.path.isdir(
            os.path.join(self.modelpath, self.model)
        ):
            inputs = self.tokenizer_instance(prompt, return_tensors="pt")

            # Generate response
            max_length = 1000
            generate_ids = self.model_instance.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
            )
            response = self.tokenizer_instance.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            return response
        else:
            return "\n Model not available"

    def parse_agent_response(self, text: str):
        """Parsing through the agent response for the  answer"""
        code = output = answer = ""

        code_match = re.search(r"Code:\s*```(?:python)?\n?(.*?)```", text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()

        output_match = re.search(r"Output:\s*(.*?)\n(?:Answer:|$)", text, re.DOTALL)
        if output_match:
            output = output_match.group(1).strip()

        answer_match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()

        return {"code": code, "output": output, "answer": answer}

    def build_agent_prompt(self, task_prompt: str) -> str:
        """Set up the prompt for the agent"""
        # Extract vectors using regex
        vector1_match = re.search(r"Vector 1: ([\d\s\.\-Ee]+)", task_prompt)
        vector2_match = re.search(r"Vector 2: ([\d\s\.\-Ee]+)", task_prompt)

        if not vector1_match or not vector2_match:
            raise ValueError("Could not extract vectors from prompt.")

        vector1 = "[" + ", ".join(vector1_match.group(1).strip().split()) + "]"
        vector2 = "[" + ", ".join(vector2_match.group(1).strip().split()) + "]"

        # Extract the actual question
        question_start = task_prompt.find("Is it more probable")
        question = task_prompt[question_start:].strip()

        # Instructions
        instructions = (
            "You are a Python reasoning assistant with access to the tool: run_code.\n"
            "You MUST strictly follow the reasoning format below, step-by-step. Do not skip or rearrange steps.\n\n"
            "To solve the task, you must:\n"
            "1. Write Python code to compare the vectors\n"
            "2. Execute it using run_code\n"
            "3. Reflect on the output\n"
            "4. Provide your Final Answer (A, B, or C)\n"
            "5. Then output a JSON block on a new line summarizing your answer and reasoning\n\n"
            " You MUST follow this exact format:\n"
            "Thought: [reasoning]\n"
            "Action: run_code\n"
            "Action Input:\n"
            "[START_CODE]\n"
            "# your code here\n"
            "[END_CODE]\n"
            "Observation: [result of the code]\n"
            "Thought: [reflect on what the result means]\n"
            "Final Answer: [A, B, or C]\n\n"
            "You must end your response with this JSON block. Do not include anything after it.\n"
            "The `answer` field must match your Final Answer exactly.\n\n"
            "```json\n"
            "{\n"
            '  "answer": "[A, B, or C]",\n'
            '  "explanation": "Brief explanation based on your reasoning."\n'
            "}\n"
            "```"
        )

        # Few-shot example
        few_shot = (
            "Example:\n"
            "vector1 = [10, 12, 14, 15]\n"
            "vector2 = [5, 6, 7, 8]\n\n"
            "Question: Is it more probable that a sample from vector1 is greater than one from vector2?\n\n"
            "Thought: I will compare their means using Python.\n"
            "Action: run_code\n"
            "Action Input:\n"
            "[START_CODE]\n"
            "import numpy as np\n"
            "vector1 = [10, 12, 14, 15]\n"
            "vector2 = [5, 6, 7, 8]\n"
            "print(np.mean(vector1) > np.mean(vector2))\n"
            "[END_CODE]\n\n"
            "Observation: True\n\n"
            "Thought: Since the mean of vector1 is greater, it is more probable.\n"
            "Final Answer: A\n\n"
            "```json\n"
            "{\n"
            '  "answer": "A",\n'
            '  "explanation": "The mean of vector1 is higher, indicating greater probability."\n'
            "}\n"
            "```"
        )

        # Combine all parts
        return (
            f"{few_shot}\n\n{instructions}"
            f"vector1 = {vector1}\nvector2 = {vector2}\n\n"
            f"Question: {question}"
        )


class ToolRegistry:
    """Tools for agent use"""

    @staticmethod
    def run_code_func(code: str) -> str:
        """Execute the code the agent writes"""
        try:
            # Extract content between [START_CODE] and [END_CODE]
            match = re.search(r"\[START_CODE\](.*?)\[END_CODE\]", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                return "Error: Missing [START_CODE] and [END_CODE] tags."

            local_vars = {}
            exec(code, {}, local_vars)  # pylint: disable=exec-used
            return f"Success. Variables: {list(local_vars.keys())}"
        except Exception:
            import traceback  # pylint: disable=import-outside-toplevel

            return f"Error:\n{traceback.format_exc()}"

    @staticmethod
    def get_tools():
        """Defines the tool for running agent code"""
        return [
            Tool(
                name="run_code",
                func=ToolRegistry.run_code_func,
                description="Executes Python code and returns variable names or errors.",
            )
        ]
