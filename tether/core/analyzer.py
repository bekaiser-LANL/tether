""" Tools for analyzing saved benchmarks """
import gc  # for efficient RAM use
import json
import os
import re

import numpy as np
import requests

from tether.core.utils import (
    create_missing_directory,
    detect_duplicate_tables,
    get_model_and_indices,
    get_parser,
)

DATA_PATH = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")
AI_GRADER_MODEL = "phi4"  # "granite3.2"
AI_GRADER_API = "ollama"  # "openai"


def truncate_response(response, num_start=3, num_end=3):
    """
    Truncate a long string response for logging. If response is structured (dict),
    return a compact summary instead.
    """
    # If it's a dict (like a parsed JSON), don't truncate — return summary
    if isinstance(response, dict):
        return f'Answer: {response.get("answer")}, Explanation: {response.get("explanation", "")[:80]}...'

    # If it's a string, truncate long responses
    if isinstance(response, str):
        lines = response.strip().splitlines()
        total = len(lines)
        if total <= num_start + num_end:
            return '\n'.join(lines)
        start = lines[:num_start]
        end = lines[-num_end:]
        return '\n'.join(start + ['... (omitted middle lines) ...'] + end)

    # If it's something else, just stringify it
    return str(response)

def extract_output(response):
    """Safely extracts output text from agent response dicts."""
    if isinstance(response, dict):
        return response.get("output", "")
    return str(response) if response is not None else ""


def extract_boolean_result_from_response(response: str) -> bool | None:
    """
    Compares the 'answer' field in a response dict to the solution (A/B/C).
    Returns True if they match, False if not, and None if 'answer' is missing.
    """
    if not isinstance(response, dict):
        return None  # not a valid JSON-style dict

    answer = response.get("answer", "").strip().upper()
    expected = solution.strip().upper()

    if not answer:
        return None  # no answer found

    return answer == expected

class Analyzer:
    """Tools for analyzing saved benchmarks"""

    def __init__(self, npz_filename, **kwargs):
        """The benchmark name is the full name of the .npz file
        without the suffix"""
        self.agent_flag = False
        parts = get_model_and_indices(npz_filename)
        if len(parts) == 4:
            self.exam_name = get_model_and_indices(npz_filename)[0]
            self.exam_idx = get_model_and_indices(npz_filename)[1]
            self.model = get_model_and_indices(npz_filename)[2]
            self.run_idx = get_model_and_indices(npz_filename)[3]
        elif len(parts) == 5:
            if get_model_and_indices(npz_filename)[2] == "agent":
                self.exam_name = get_model_and_indices(npz_filename)[0]
                self.ci_method = get_model_and_indices(npz_filename)[1]
                self.model = get_model_and_indices(npz_filename)[3]
                self.run_idx = get_model_and_indices(npz_filename)[4]
            else:
                self.exam_name = get_model_and_indices(npz_filename)[0]
                self.ci_method = get_model_and_indices(npz_filename)[1]
                self.exam_idx = get_model_and_indices(npz_filename)[2]
                self.model = get_model_and_indices(npz_filename)[3]
                self.run_idx = get_model_and_indices(npz_filename)[4]
        self.verbose = kwargs.get("verbose", False)
        self.grade_estimate = kwargs.get("grade_estimate", False)
        self.human_review = kwargs.get("human_review", False)
        self.print_vars = kwargs.get("print_vars", False)
        self.print_responses = kwargs.get("print_responses", False)
        self.completed_path = os.path.join(DATA_PATH, "completed")  # ,self.model)
        self.completed_path = os.path.join(data_path, 'completed')#,self.model)
        self.grader_llm = kwargs.get("grader_model", self.model)
        self.grader_model = self.load_llm(self.grader_llm)
        self.exam_name = self.exam_name
        if self.agent_flag:
            self.json_path = get_json_filename(
                self.completed_path,
                self.exam_name+ '_' + self.ci_method,
                self.exam_idx,
                self.model,
                self.agent_flag
            )
        self.npz_filepath = os.path.join(
              self.completed_path,
              npz_filename + '.npz'
        )
        self.npz_filename = npz_filename
        self.graded_benchmark_path = os.path.join(data_path,'graded')
        create_missing_directory(self.graded_benchmark_path)
        self.graded_benchmark_by_model_path = os.path.join(
            self.graded_benchmark_path,
            self.model
        )        
        create_missing_directory(self.graded_benchmark_by_model_path)

        # list of ABC multiple choice benchmarks:
        self.abc_multiple_choice_list = [
            "MediatedCausality",
            "MediatedCausalitySmoking",
            "MediatedCausalityWithMethod",
            "SimpleInequality",
            "SimpleInequalityWithMethod",
        ]


        if self.agent_flag:
            with open(self.json_path) as f:
                self.json_responses = json.load(f)

        if self.print_vars:
            # --print_vars
            self.data = np.load(self.npz_filepath, allow_pickle=True)
            self.print_keys()

        if self.print_responses:
            # --print_responses
            self.data = np.load(self.npz_filepath, allow_pickle=True)
            self.print_completed_benchmark()

        if self.grade_estimate:
            # --grade_estimate
            self.data = np.load(self.npz_filepath, allow_pickle=True)
            self.provisional_grade_with_ai()

        if self.human_review:
            # --human_review
            self.final_grade_by_human()

    def final_grade_by_human(self):
        """Assign the final grade"""
        prov_file = self.npz_filename + "_provisional_grade.npz"
        final_file = self.npz_filename + "_final_grade.npz"
        open_path = os.path.join(self.graded_benchmark_by_model_path, prov_file)
        save_path = os.path.join(self.graded_benchmark_by_model_path, final_file)
        if not os.path.exists(open_path):
            raise FileNotFoundError(f"File not found: {open_path}")
        loaded = np.load(open_path, allow_pickle=True)
        self.data = {key: loaded[key].copy() for key in loaded.files}
        idx = self.get_true_indices(self.data["human_review"])
        n_review = len(idx)
        print("\n\n")
        for k in range(0, n_review):
            print("********************************************************")
            print("\n\nLLM response:\n--------------------------------\n")
            print(self.data["responses"][idx[k]])
            print("\n--------------------------------")
            # ,np.round(1.-k/n_review,1)*100)
            print(f"\n{n_review-k} remaining to review ")
            print("Solution: ", self.data["solution"][idx[k]])
            human = input(
                "Is the LLM response correct? Answer `y' or `n' for yes "
                "and no, respectively. Press any other key to skip.\n"
            )
            if human == "n":
                self.data["grade_estimate"][idx[k]] = False
                self.data["human_review"][idx[k]] = False
                print(
                    "You said incorrect. Therefore, "
                    f"grade_estimate = {self.data['grade_estimate'][idx[k]]}"
                )
            elif human == "y":
                self.data["grade_estimate"][idx[k]] = True
                self.data["human_review"][idx[k]] = False
                print(
                    "You said correct. Therefore, "
                    f"grade_estimate = {self.data['grade_estimate'][idx[k]]}"
                )
            print(
                "You responded y or n. Therefore "
                f"human review flag = {self.data['human_review'][idx[k]]}"
            )
            print("\n\n\n********************************************************")

        # save the final grades:
        if np.sum(self.data["human_review"]) == 0:
            graded_npz_filename = self.npz_filename + "_final_grade.npz"
        else:
            graded_npz_filename = self.npz_filename + "_provisional_grade.npz"
        save_path = os.path.join(
            self.graded_benchmark_by_model_path, graded_npz_filename
        )
        np.savez(save_path, **self.data)

    def get_true_indices(self, boolean_array):
        """Get array indices"""
        return np.where(boolean_array)[0]

    def print_keys(self):
        """List all keys stored in the file"""
        print("\n Keys:\n", self.data.files)

    def print_completed_benchmark(self):
        """Print the completed benchmark Q&A"""
        # n_problems = len(self.data["question"])
        for i in range(0, 2):
            print("\n\n******************************************************")
            print("\n question = ", self.data["question"][i])
            print(" responses = ", self.data["responses"][i])
            print(" solution = ", self.data["solution"][i])
            print("\n")
            # print(" unbiased solution = ",data["unbiased_solution"][i])
            # print(" biased solution = ",data["biased_solution"][i])

    def verify_no_duplicates(self):
        """Check for duplicate questions"""
        if self.exam_name.startswith("MediatedCausality"):
            has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(
                self.data["table"]
            )
            print(
                f"\n Benchmark: {self.exam_name}"
                f"\n Duplicate tables detected: {has_duplicates}"
                f"\n Number of problems: {n_problems}"
            )
            if has_duplicates:
                print(f" {duplicate_pairs} duplicate pairs found")
        print(
            f"\n Verify no duplicate problems needs to be implemented for {self.exam_name}"
        )

    def ask_openai(self, question, client, model_choice):
        """Method for prompting & recording OpenAI products"""
        openai_reasoning_model_list = ["o3-mini", "o1", "o3"]
        openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
        if model_choice in openai_classic_model_list:
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    temperature=0.0,  # 0.0 (deterministic) to 1.0 (random)
                )
                if response.choices[0].message.content == "True":
                    return True
                if response.choices[0].message.content == "False":
                    return False
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        elif model_choice in openai_reasoning_model_list:
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question},
                    ],
                    reasoning_effort="high",  # Options: "low", "medium", "high")
                )
                if response.choices[0].message.content.strip() == "True":
                    return True
                if response.choices[0].message.content.strip() == "False":
                    return False
            except Exception as e:  # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")

    def contains_connection_error(self, response):
        """Test the connection"""
        # Safely extract the relevant text from the dict
        if isinstance(response, dict):
            text = response.get("output") or response.get("response") or ""
        else:
            text = str(response) if response is not None else ""
        pattern = r"\bError: Connection error\."
        return re.search(pattern, text) is not None

    def ask_ollama(self, prompt, model):
        """Interact with the ollama models"""
        response = None
        # This will work — as long as you have "ollama serve" running
        # in one terminal and the model is on the list.
        # ensure_ollama_running()
        url = "http://localhost:11434/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}
        # Send the request to the API
        request = requests.post(url, json=payload, timeout=120)
        if request.status_code == 200:
            # This is the standard HTTP status code for a successful request.
            # Successful response from the Ollama API
            response = request.json()["response"]
        else:
            print("Error:", request.status_code, request.text)
        return response

    def load_llm(self, model_name):
        if model_name in openai_all_model_list:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=0)
        elif model_name in ollama_model_list:
            from langchain_ollama import ChatOllama
            return ChatOllama(model=model_name)
        elif model_name in anthropic_model_list:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=model_name)
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    def build_grading_chain(self):
        return PromptTemplate.from_template(
            "You are a grading assistant.\n"
            "You will receive the correct answer and the AI's JSON output, which includes an 'answer' key.\n\n"
            "Compare the AI's 'answer' value to the correct one. Ignore any missing 'explanation'.\n\n"
            "Correct answer: {solution}\n"
            "AI JSON: {ai_json}\n\n"
            "Is the AI's answer correct?\n"
            "Respond ONLY in this JSON format:\n"
            '{{"result": true, "explanation": "Answers match."}}'
        ) | self.grader_model

    def provisional_grade_with_ai(self):
        """Estimate the grade with openai and deterministic pattern"""
        broken_flag = False
        n_problems = len(self.data["question"])
        grade = np.full(n_problems, True)
        # assume correct until proven otherwise
        human_review = np.full(n_problems, False)

        self.grading_chain = self.build_grading_chain()

        for j in range(0, n_problems):
            if self.contains_connection_error(self.data["responses"][j]):
                broken_flag = True
            if self.exam_name in self.abc_multiple_choice_list:

              if self.agent_flag:
                    response = self.json_responses[j]
                else:
                    response = self.data["responses"][j]

                # Create structured input
                solution = self.data["solution"][j]
                grader_input = {
                    "solution": solution,
                    "ai_json": response
                }

                # Run LangChain grader
                grading_response = self.grading_chain.invoke(grader_input)
                grading_text = grading_response.content if hasattr(grading_response, "content") else str(grading_response)

                # Parse result
                try:
                    if isinstance(grading_text, str):
                        match = re.search(r'```json\s*(\{.*?\})\s*```', grading_text, re.DOTALL)
                        grading_json_str = match.group(1) if match else grading_text.strip()
                        parsed_response = json.loads(grading_json_str)
                    else:
                        parsed_response = grading_text  # already a dict

                    result = parsed_response.get("result", None)
                    ai_grader = bool(result) if result is not None else False
                except Exception as e:
                    print(f"Grader JSON parsing failed for question {j}: {e}")
                    print("Raw grader output:", grading_text)
                    ai_grader = False
                    human_review[j] = True

                gc.collect() # for efficient RAM use

                deterministic_grader = self.deterministic_grader_abc(
                    self.data["solution"][j], self.data["responses"][j]
                )
            elif self.exam_name == "StandardDeviation":
                print(" StandardDeviation needs to be set up with ")
                print(" two prompts for ai grader and two solutions")
            else:
                print(" Grader not set up")
            if ai_grader == deterministic_grader:
                grade[j] = deterministic_grader  # answer is correct or not
            else:
                human_review[j] = True  # flag for human review
            if self.verbose:
                print("\n\n**************************************************")
                print("\n", self.exam_name)
                print(" question: ", j)
                # print(" llm response: ",self.data["responses"][j])
                print(" truncated llm response: ", truncated_response)
                print(" correct answer: ", self.data["solution"][j])
                print(" AI grader: is it correct? ", ai_grader)
                print(" deterministic grader: is it correct? ", deterministic_grader)
                print(" correct answer? ", grade[j])
                print(" human needed? ", human_review[j])

        if self.verbose:
            print("\n\n Total score (%): ", np.sum(grade) / n_problems * 100.0)
            print(" Number of questions needing review: ", np.sum(human_review))
            print("\n ")

        # Create a dictionary with existing arrays
        all_arrays = {key: self.data[key] for key in self.data.files}
        # Add your new array
        all_arrays["grade_estimate"] = grade
        all_arrays["human_review"] = human_review
        # # Save everything to a new .npz file:
        if broken_flag:
            graded_npz_filename = self.npz_filename + "_RERUN_THIS_BENCHMARK.npz"
        elif np.sum(human_review) == 0:
            graded_npz_filename = self.npz_filename + "_final_grade.npz"
        else:
            graded_npz_filename = self.npz_filename + "_provisional_grade.npz"
        save_path = os.path.join(
            self.graded_benchmark_by_model_path, graded_npz_filename
        )
        np.savez(save_path, **all_arrays)

    def deterministic_grader_abc(self, solution, response, choices=["A", "B", "C"]):
        """
        Checks if the correct multiple-choice answer is found in the response.
        Returns True if the correct answer is detected, False otherwise.
        """
        if solution not in choices:
            raise ValueError(f"Invalid solution '{solution}', must be one of {choices}")

        # If structured JSON, extract directly
        if isinstance(response, dict):
            answer = response.get("answer", None)
            if isinstance(answer, str):
                return answer.strip().upper() == solution.strip().upper()
            else:
                return False

        if not isinstance(response, str):
            return False

        # Fallback for unstructured text (legacy logic)
        text = extract_output(response)
        last_line = text.strip().splitlines()[-1] if text.strip() else ""

        # Try to extract from "Answer: C" or "Answer: C,"
        match = re.search(r"Answer:\s*([A-Ca-c])[,\.]?", text, re.IGNORECASE)
        if match:
            return match.group(1).upper() == solution.upper()

        # Fallback: just try last standalone letter
        match = re.search(r"\b([A-Ca-c])\b", last_line)
        if match:
            return match.group(1).upper() == solution.upper()

        return False

def main():
    """Analyze the benchmark"""
    parser = get_parser(script="analyze")
    args = parser.parse_args()
    kwargs = vars(args)
    npz_filename = kwargs.pop("exam_name")
    verbose = kwargs.pop("verbose", False)

    Analyzer(npz_filename, verbose=verbose, **kwargs)


if __name__ == "__main__":
    main()
