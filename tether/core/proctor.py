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
        self.model = model
        self.agent_flag = kwargs.get("agent", False)
        exam_only = get_model_and_indices(self.exam_name)[0]
        if self.agent_flag:
            if self.model in ollama_model_list:
                self.llm = ChatOllama(model=self.model)
            elif self.model in openai_all_model_list:
                self.llm = ChatOpenAI(model=self.model)
            else:
                raise ValueError(f"Unsupported model type.")
            if exam_only == 'SimpleInequality' or exam_only == 'SimpleInequalityWithMethod':
                self.code_writer_prompt = self.build_code_writer_prompt()
                self.answer_extractor_prompt = self.build_answer_extractor_prompt()

                self.code_writer_chain = self.build_code_writer_prompt() | self.llm
                self.answer_extractor_chain = self.build_answer_extractor_prompt() | self.llm
                self.code_executor = PythonAstREPLTool().run
            else:
                self.code_writer_prompt_causal = self.build_code_writer_prompt_causal()
                self.answer_extractor_prompt = self.build_answer_extractor_prompt()

                self.code_writer_chain_causal = self.code_writer_prompt_causal | self.llm
                self.answer_extractor_chain = self.build_answer_extractor_prompt() | self.llm
                self.code_executor = PythonAstREPLTool().run

        else:
            self.agent_flag = False
        self.verbose = kwargs.get("verbose", False)
        self.modelpath = kwargs.get("model_path")
        self.results_path = os.path.join(self.benchmark_path, "completed")
        self.client = None
        self.npz_filename = get_npz_filename(
            self.results_path,
            self.exam_name ,
            self.exam_idx,
            self.model,
            self.agent_flag
        )
        create_missing_directory(self.results_path)
        create_missing_directory(self.saved_benchmark_path)

        benchmark = load_saved_benchmark(
            self.saved_benchmark_path, self.exam_name, self.exam_idx
        )
        self.questions = benchmark["question"]
        # Load any existing file if restarting
        self.json_path = get_json_filename(
        self.results_path,
        self.exam_name,
        self.exam_idx,
        self.model,
        self.agent_flag
        )

        if os.path.exists(self.json_path):
            with open(self.json_path) as f:
                responses = json.load(f)
        else:
            responses = []
        answers = self.give_benchmark(benchmark, responses=responses)
        np.savez(self.npz_filename, **benchmark, responses=answers)
        print(f"Saved completed benchmark output to npz: {self.npz_filename}")

        print(f"Saved completed benchmark output to JSON: {self.json_path}")
        
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
        if responses is None:
            responses = []
        model = get_model_and_indices(self.exam_name)[0]
        for i in range(len(responses), n):#0,n):
            if i < len(responses):
                if self.verbose:
                    print(f"Skipping question {i}, already saved.")
                continue
            prompt = benchmark["question"][i]
            if self.verbose:
                print('\n Question ',i)
                print(prompt)
            if self.agent_flag:
                if model == 'SimpleInequality' or model == 'SimpleInequalityWithMethod':
                    print('in here')
                    vector1, vector2, question = self.parse_vectors_and_question(prompt)
                    response = self.run_and_extract_answer(vector1, vector2, question, i)
                else:
                    response = self.run_and_extract_answer_causal(prompt, i)
            else:
                response = self.give_question_to_llm(prompt)
                print(response)
            responses.append(response)
            if self.verbose:
                print(f"Saved question {i} to {self.json_path}")

            with open(self.json_path, "w") as f:
                json.dump(responses, f, indent=2)

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
    def parse_vectors_and_question(self, task_prompt: str):
        """Extract vector1, vector2, and the question from the raw text."""
        vector1_match = re.search(r"Vector 1: ([\d\s\.\-Ee]+)", task_prompt)
        vector2_match = re.search(r"Vector 2: ([\d\s\.\-Ee]+)", task_prompt)
        question_start = task_prompt.find("Is it more probable")
        if not (vector1_match and vector2_match and question_start != -1):
            raise ValueError("Failed to extract vectors or question.")

        vector1 = "[" + ", ".join(vector1_match.group(1).strip().split()) + "]"
        vector2 = "[" + ", ".join(vector2_match.group(1).strip().split()) + "]"
        question = task_prompt[question_start:].strip()
        return vector1, vector2, question

    def run_and_extract_answer(self, vector1, vector2, question, i):
        max_retries = 10
        execution_success = False
        execution_result = ""
        code_string = ""
        code_text = ""
        code_response = None

        for attempt in range(max_retries):
            print(f"\n Attempt {attempt+1} — Generating code for question {i}...")

            #Generate code with LLM
            code_response = self.code_writer_chain.invoke({
                "vector1": vector1,
                "vector2": vector2,
                "question": question
            })

            #Extract raw text
            code_text = code_response.content if hasattr(code_response, "content") else str(code_response)

            #Extract code block from tags
            match = re.search(r"\[START_CODE\](.*?)\[END_CODE\]", code_text, re.DOTALL)
            code_string = match.group(1).strip() if match else code_text

            print("Generated code:\n", code_string)

            #Attempt execution
            execution_result = self.code_executor(code_string)
            execution_output_str = str(execution_result)

            print("Execution output:\n", execution_output_str)

            #Check if execution succeeded
            error_keywords = [
                "Traceback", "NameError", "TypeError", "ValueError",
                "IndexError", "ZeroDivisionError", "SyntaxError"
            ]
            if not any(err in execution_output_str for err in error_keywords):
                execution_success = True
                break
            else:
                print("Code execution failed. Retrying...")

            valid_answers = {"A", "B", "C"}
            if execution_output.strip() not in valid_answers:
                print("Code did not return a valid answer. Retrying...")
                retry()
            else:
                answer = execution_output.strip()
        #Final fallback if all attempts fail
        if not execution_success:
            # Save last attempted code to file
            fail_dir = os.path.join(self.results_path, "failed_code")
            os.makedirs(fail_dir, exist_ok=True)

            fail_path = os.path.join(fail_dir, f"q{i}_failed.py")
            with open(fail_path, "w") as f:
                f.write("# Failed code generation after retries\n")
                f.write("# Vector 1:\n")
                f.write(f"{vector1}\n")
                f.write("# Vector 2:\n")
                f.write(f"{vector2}\n\n")
                f.write("# Last generated code:\n")
                f.write(code_string)

            print("All attempts failed. Skipping this question.")
            return {
                "code": code_string,
                "execution_result": execution_result,
                "final_answer": None,
                "explanation": "Code failed to execute successfully after multiple generations."
            }

        #Extract final answer
        extractor_input = {
            "execution_output": execution_result,
            "question": question
        }
        extracted_raw = self.answer_extractor_chain.invoke(extractor_input)
        raw_text = extracted_raw.content if hasattr(extracted_raw, "content") else str(extracted_raw)

        # Remove code block formatting if needed
        raw_text = re.sub(r"```json|```", "", raw_text).strip()

        # Extract JSON
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        json_str = match.group(0) if match else raw_text

        #extracted_json = json.loads(json_str)
        match = re.search(r'\{.*?\}', json_str, re.DOTALL)
        if match:
            try:
                extracted_json = json.loads(match.group(0))
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode failed: {e}")
                extracted_json = {"answer": None, "explanation": f"Bad JSON: {e}"}
        else:
           print("❌ No JSON object found in output.")
           extracted_json = {"answer": None, "explanation": "No JSON found"}

        print(extracted_json)
        return extracted_json

    def parse_causal_prompt(self, prompt: str):
        """
        Extracts X, Y, Z variable names, 8 numerical table values, and the question from a causal inference prompt.
        """
        # Extract the table counts (expect 8 of them)
        count_pattern = r'is (\d+)\.|\b(\d+) samples'
        counts = [int(m[0] or m[1]) for m in re.findall(count_pattern, prompt)]
        if len(counts) != 8:
            print(f"⚠️ Expected 8 counts, but found {len(counts)}")

        # Extract X, Y, Z variable names
        x_match = re.search(r"do not (\w+), do not", prompt)
        y_match = re.search(r"do not \w+, do not (\w+),", prompt)
        z_match = re.search(r"and do not (\w+)", prompt)

        x_name = x_match.group(1) if x_match else None
        y_name = y_match.group(1) if y_match else None
        z_name = z_match.group(1) if z_match else None

        # Extract the final causal question
        question_match = re.search(r"Does (.*?)\?", prompt)
        causal_question = question_match.group(0).strip() if question_match else None

        return {
             "x_name": x_name,
             "y_name": y_name,
             "z_name": z_name,
             "counts": counts,
             "question" : causal_question
        }

    def run_and_extract_answer_causal(self, question, i):
        """
        Agent-based causal inference pipeline:
        1. Generate code to solve the causal question
        2. Execute code and get result
        3. Use an LLM to extract answer (A/B/C) and explanation from execution result
        """
        parsed = self.parse_causal_prompt(question)
        flattened_input = {
            "x_name": parsed["x_name"],
            "y_name": parsed["y_name"],
            "z_name": parsed["z_name"],
            "question": parsed["question"],
            "count0": parsed["counts"][0],
            "count1": parsed["counts"][1],
            "count2": parsed["counts"][2],
            "count3": parsed["counts"][3],
            "count4": parsed["counts"][4],
            "count5": parsed["counts"][5],
            "count6": parsed["counts"][6],
            "count7": parsed["counts"][7]
        }
        execution_success = False
        for attempt in range(10):
            if self.verbose:
                print(f"\n Attempt {attempt + 1} — Generating code for question {i}...")

            try:
                # Step 1: Generate Python code
                code_response = self.code_writer_chain_causal.invoke(flattened_input)
                code_string = code_response.content if hasattr(code_response, "content") else str(code_response)

                if self.verbose:
                    print("Generated code:\n", code_string)

                match = re.search(r"\[START_CODE\](.*?)\[END_CODE\]", code_string, re.DOTALL)
                if match:
                    code_string = match.group(1).strip()
                else:
                    print("No [START_CODE] ... [END_CODE] block found.")
                    code_string = ""  # or raise an error

                # Step 2: Execute code
                execution_result = self.code_executor(code_string)
                output = execution_result.strip()

                if self.verbose:
                    print("Execution output:\n", execution_result)

                # Step 3: Validate execution output (basic error filter)
                error_keywords = ["Traceback", "NameError", "TypeError",
                                  "ValueError", "ModuleNotFoundError",
                                  "Exception", "SyntaxError"]
                if any(err in execution_result for err in error_keywords):
                    print("Code execution failed. Retrying...\n")
                    continue

                valid_answers = {"A", "B", "C"}
                if output in valid_answers:
                    execution_success = True
                    break
                else:
                    print("Code did not return valid answer. Retrying...\n")
                    continue

            except Exception as e:
                print(f"Exception during code generation or execution: {e}")
        #Final fallback if all attempts fail
        if not execution_success:
            # Save last attempted code to file
            fail_dir = os.path.join(self.results_path, "failed_code")
            os.makedirs(fail_dir, exist_ok=True)

            fail_path = os.path.join(fail_dir, f"q{i}_failed.py")
            with open(fail_path, "w") as f:
                f.write("# Failed code generation after retries\n")
                f.write("# Last generated code:\n")
                f.write(code_string)

            print("All attempts failed. Skipping this question.")
            return {
                "code": code_string,
                "execution_result": execution_result,
                "final_answer": None,
                "explanation": "Code failed to execute successfully after multiple generations."
            }

        # Step 4: Extract final answer using LLM
        extractor_input = {
            "execution_output": execution_result,
            "question": parsed["question"]
        }

        try:
            extracted_raw = self.answer_extractor_chain.invoke(extractor_input)
            raw_text = extracted_raw.content if hasattr(extracted_raw, "content") else str(extracted_raw)

            # Try parsing JSON
            match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if match:
                extracted_json = json.loads(match.group(0))
            else:
                extracted_json = {
                    "answer": None,
                    "explanation": "Could not find valid JSON block in response."
                }

        except Exception as e:
            extracted_json = {
                "answer": None,
                "explanation": f"Exception during extraction: {e}"
            }

        print(extracted_json)
        return extracted_json
    def build_code_writer_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            "You are a Python assistant. Given vector1, vector2, and a question, "
            "output only Python code between [START_CODE] and [END_CODE] that answers the question.\n"
            "You must extract and define `vector1` and `vector2` based on the text of the prompt.\n"
            "Do not explain.\n\n"
            "vector1 = {vector1}\n"
            "vector2 = {vector2}\n"
            "Question: {question}"
        )

    def build_code_writer_prompt_causal(self) -> PromptTemplate:
        return PromptTemplate(
        input_variables=[
            "x_name", "y_name", "z_name",
            "count0", "count1", "count2", "count3",
            "count4", "count5", "count6", "count7",
            "question"
        ],
        template="""
        You are a Python data scientist. Your task is to assess whether {x_name} causes {y_name} using a stratified observational dataset.

        The data is stratified based on the binary variables {x_name}, {y_name}, and {z_name}, and is represented as 8 cell counts in the following format:

        index | {x_name} | {y_name} | {z_name} | count
        ------|----------|----------|----------|------
          0   |    0     |    0     |    0     | {count0}
          1   |    0     |    0     |    1     | {count1}
          2   |    0     |    1     |    0     | {count2}
          3   |    0     |    1     |    1     | {count3}
          4   |    1     |    0     |    0     | {count4}
          5   |    1     |    0     |    1     | {count5}
          6   |    1     |    1     |    0     | {count6}
          7   |    1     |    1     |    1     | {count7}

        Use only built-in Python, numpy, or scipy to determine causality between {x_name} and {y_name} while controlling for {z_name}.

        At the end, print:
        - "A" if the data supports causality,
        - "B" if it supports no effect,
        - "C" if it's inconclusive.

        Wrap your code in [START_CODE] and [END_CODE]. Do not print anything else.

        Here is the question:
        {question}
        """
        )

    def build_answer_extractor_prompt(self) -> PromptTemplate:
        return PromptTemplate.from_template(
            "You are a statistics assistant. You are given a multiple-choice question "
            "and the output from a Python program that already chose an answer (A, B, or C).\n\n"
            "You MUST return exactly the letter printed by the code as the `answer` field.\n\n"
            "Do not reinterpret the result. Just reflect what the code printed.\n\n"
            "Question:\n{question}\n\n"
            "Output:\n{execution_output}\n\n"
            "Respond ONLY in strict JSON format:\n"
            '{{"answer": "C", "explanation": "Because the output of the code was C."}}'
        )

class ToolRegistry:
    """Tools for agent use"""
    @staticmethod
    def run_code_func(code: str) -> str:
        """Execute the code the agent writes"""
        print("Code received by run_code:")
        print(code)
        try:
            # Extract content between [START_CODE] and [END_CODE]
            match = re.search(r"\[START_CODE\](.*?)\[END_CODE\]", code, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                return "Error: Missing [START_CODE] and [END_CODE] tags."

            local_vars = {}

            # Capture stdout
            import io, contextlib
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                exec(code, local_vars, local_vars)

            # Prefer result variable if it exists
            if "result" in local_vars:
                return str(local_vars["result"])

            # Otherwise return captured print output
            output = buffer.getvalue().strip()
            return output if output else "Code executed successfully, no output."

        except Exception:
            import traceback
            return f"Error:\n{traceback.format_exc()}"

    @staticmethod
    def get_tools():
        """Defines the tool for running agent code"""
        return [
            Tool(
                name="run_code",
                func=ToolRegistry.run_code_func,
                description="Executes Python code and returns the result or printed output."
            )
        ]
