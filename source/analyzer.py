""" Tools for analyzing saved benchmarks """
import os
import re
import requests
import gc # for efficient RAM use
import json
import numpy as np
from source.utils import get_model_and_indices, create_missing_directory
from source.utils import detect_duplicate_tables

data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

ai_grader_model = 'phi4' # 'granite3.2'
ai_grader_api = 'ollama' # 'openai'

def truncate_response(text, num_start=3, num_end=3):
    """
    Prints the first `num_start` and last `num_end` lines of a long string.
    """
    lines = text.strip().splitlines()
    total = len(lines)

    if total <= num_start + num_end:
        return '\n'.join(lines)  # Return full text if it's already short

    start = lines[:num_start]
    end = lines[-num_end:]

    return '\n'.join(start + ['... (omitted middle lines) ...'] + end)

def extract_boolean_result_from_response(response: str) -> bool | None:
    """
    Extracts the boolean value of 'result' from a JSON block in the LLM response.
    Returns True or False if found, or None if parsing fails.
    """
    match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
    if match:
        try:
            json_str = match.group(1)
            data = json.loads(json_str)
            return data.get("result")
        except json.JSONDecodeError:
            return None
    return None

class Analyzer():
    """ Tools for analyzing saved benchmarks """

    def __init__(self, npz_filename, **kwargs):
        """ The benchmark name is the full name of the .npz file 
        without the suffix """

        self.ai_grader_api = ai_grader_api

        parts = get_model_and_indices(npz_filename)
        if len(parts) == 4:
            self.exam_name = get_model_and_indices(npz_filename)[0]
            self.exam_idx  = get_model_and_indices(npz_filename)[1]       
            self.model     = get_model_and_indices(npz_filename)[2]
            self.run_idx   = get_model_and_indices(npz_filename)[3]            
        elif len(parts) == 5:
            self.exam_name = get_model_and_indices(npz_filename)[0]
            self.ci_method = get_model_and_indices(npz_filename)[1]
            self.exam_idx  = get_model_and_indices(npz_filename)[2]       
            self.model     = get_model_and_indices(npz_filename)[3]
            self.run_idx   = get_model_and_indices(npz_filename)[4]
        self.verbose = kwargs.get('verbose', False)
        self.grade_estimate = kwargs.get('grade_estimate', False)
        self.human_review = kwargs.get('human_review', False)
        self.print_vars = kwargs.get('print_vars', False)
        self.print_responses = kwargs.get('print_responses', False)
        self.completed_path = os.path.join(data_path, 'completed',self.model)
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
        self.ABC_multiple_choice_list = [
            'MediatedCausality',
            'MediatedCausalitySmoking',
            'MediatedCausalityWithMethod',
            'SimpleInequality',
            'SimpleInequalityWithMethod'
        ]
        
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
        """ Assign the final grade """
        prov_file = self.npz_filename + '_provisional_grade.npz'
        final_file = self.npz_filename + '_final_grade.npz' 
        open_path = os.path.join(self.graded_benchmark_by_model_path, prov_file)
        save_path = os.path.join(self.graded_benchmark_by_model_path, final_file)
        if not os.path.exists(open_path):
            raise FileNotFoundError(f"File not found: {open_path}")
        loaded = np.load(open_path, allow_pickle=True)
        self.data = {key: loaded[key].copy() for key in loaded.files}
        idx = self.get_true_indices(self.data["human_review"])
        n_review = len(idx)
        print('\n\n')
        for k in range(0,n_review):
            print('********************************************************')
            print('\n\nLLM response:\n--------------------------------\n')
            print(self.data["responses"][idx[k]])
            print('\n--------------------------------')
            print('\nSolution: ',self.data["solution"][idx[k]])
            human = input("Is the LLM response correct? Answer `y' or `n' for yes and no, respectively. Press any other key to skip.\n")
            if human == 'n':
                self.data["grade_estimate"][idx[k]] = False
                self.data["human_review"][idx[k]] = False
                print(f'You said incorrect. Therefore, grade_estimate = {self.data["grade_estimate"][idx[k]]}')
            elif human == 'y':
                self.data["grade_estimate"][idx[k]] = True
                self.data["human_review"][idx[k]] = False
                print(f'You said correct. Therefore, grade_estimate = {self.data["grade_estimate"][idx[k]]}')
            print(f'You responded y or n. Therefore human review flag = {self.data["human_review"][idx[k]]}')
            print('\n\n\n********************************************************')

        # save the final grades:
        if np.sum(self.data["human_review"]) == 0:
            graded_npz_filename = self.npz_filename + '_final_grade.npz'
        else:
            graded_npz_filename = self.npz_filename + '_provisional_grade.npz'
        save_path = os.path.join(self.graded_benchmark_by_model_path, graded_npz_filename)
        np.savez(save_path, **self.data)

        # # CHECK
        # loaded = np.load(save_path, allow_pickle=True)
        # print("\n Keys:\n", loaded.files)
        # print(len(self.get_true_indices(loaded["human_review"])))

    def get_true_indices(self, boolean_array):
        return np.where(boolean_array)[0]

    def print_keys(self):
            """ List all keys stored in the file """
            print("\n Keys:\n", self.data.files)

    def print_completed_benchmark(self):
        """ Print the completed benchmark Q&A """
        n_problems = len(self.data["question"])
        for i in range(0,2):
            print('\n\n******************************************************')
            print('\n question = ',self.data["question"][i])
            print(' responses = ',self.data["responses"][i])
            print(' solution = ',self.data["solution"][i])
            print('\n')
            # print(' unbiased solution = ',data["unbiased_solution"][i])
            # print(' biased solution = ',data["biased_solution"][i])

    def verify_no_duplicates(self):
        if self.exam_name.startswith('MediatedCausality'):
            has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(self.data['table'])
            print(f"\n Benchmark: {self.exam_name}"
                f"\n Duplicate tables detected: {has_duplicates}"
                f"\n Number of problems: {n_problems}")
            if has_duplicates:
                print(f" {duplicate_pairs} duplicate pairs found")   
        print(f"\n Verify no duplicate problems needs to be implemented for {self.exam_name}")

    def ask_openai(self, question, client, model_choice):
        """ Method for prompting & recording OpenAI products """
        openai_reasoning_model_list = ['o3-mini','o1','o3']
        openai_classic_model_list = ["gpt-4.5-preview", "gpt-4o", "gpt-4.1"]
        if model_choice in openai_classic_model_list:
            try:
                response = client.chat.completions.create(
                    model=model_choice,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": question}
                    ],
                    temperature=0.0 # 0.0 (deterministic) to 1.0 (random)
                )
                if response.choices[0].message.content == 'True':
                    return True 
                elif response.choices[0].message.content == 'False':
                    return False
            except Exception as e: # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        elif model_choice in openai_reasoning_model_list:
            try:
                response = client.chat.completions.create(model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                reasoning_effort='high' # Options: 'low', 'medium', 'high')
                )
                if response.choices[0].message.content.strip() == 'True':
                    return True 
                elif response.choices[0].message.content.strip() == 'False':
                    return False
            except Exception as e: # pylint: disable=broad-exception-caught
                return f"Error: {e}"
        else:
            return print("\n Model choice not available ")

    def contains_connection_error(self,text):
        """ Checks LLM responses for this common problem """
        pattern = r"\bError: Connection error\."
        return re.search(pattern, text) is not None

    def ask_ollama(self, prompt, model):
        response = None
        # This will work — as long as you have 'ollama serve' running
        # in one terminal and the model is on the list.
        #ensure_ollama_running()
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        # Send the request to the API
        request = requests.post(url, json=payload, timeout=120)
        if request.status_code == 200:
            # This is the standard HTTP status code for a successful request.
            # Successful response from the Ollama API
            response = request.json()["response"]
        else:
            print("Error:", request.status_code, request.text)
        return response

    def provisional_grade_with_ai(self):
        """ Estimate the grade with openai and deterministic pattern """

        from openai import OpenAI # pylint: disable=import-outside-toplevel
        openai_api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(api_key=openai_api_key)
        broken_flag = False

        n_problems = len(self.data["question"])
        grade = np.full(n_problems, True) 
        # assume correct until proven otherwise
        human_review = np.full(n_problems, False)
        for j in range(0,n_problems):
            if self.contains_connection_error(self.data['responses'][j]):
                broken_flag = True
            if self.exam_name in self.ABC_multiple_choice_list:
                #print('\n\n',self.data['responses'][j])
                truncated_response = truncate_response(self.data['responses'][j])
                prompt = (
                    f"The correct answer is {self.data['solution'][j]}, "
                    f"is the following response correct: {truncated_response}? "
                    "Please answer True or False, and scan the end of the response in particular. Output in the following format:\n"
                    "```json\n"
                    "{\n"
                    '  "result": true,\n'
                    '  "explanation": "Because the sample size is too small."\n'
                    "}\n"
                    "```"
                )
                if self.ai_grader_api == 'ollama':
                    json_response = self.ask_ollama(prompt, ai_grader_model)
                    ai_grader = extract_boolean_result_from_response(json_response)
                    gc.collect() # for efficient RAM use
                else:
                    ai_grader = self.ask_openai(prompt, client,ai_grader_model)
                    print('\n OPENAI MODELS NEED JSON RESPONSE CONSTRAINT')
                deterministic_grader = self.deterministic_grader_ABC(
                    self.data["solution"][j],
                    self.data["responses"][j]
                )
            elif self.exam_name == 'StandardDeviation':
                print(' StandardDeviation needs to be set up with ')
                print(' two prompts for ai grader and two solutions')
            else:
                print(' Grader not set up')    
            if ai_grader == deterministic_grader:
                grade[j] = deterministic_grader # answer is correct or not
            else:
                human_review[j] = True # flag for human review
            if self.verbose:
                print('\n\n**************************************************')
                print('\n',self.exam_name)
                print(' question: ',j)
                #print(' llm response: ',self.data["responses"][j])
                print(' truncated llm response: ',truncated_response)
                print(' correct answer: ',self.data["solution"][j])
                print(' AI grader: is it correct? ',ai_grader)
                print(' deterministic grader: is it correct? ',deterministic_grader)
                print(' correct answer? ',grade[j])
                print(' human needed? ',human_review[j])

        if self.verbose:
            print('\n\n Total score (%): ',np.sum(grade)/n_problems*100.)
            print(' Number of questions needing review: ',np.sum(human_review))
            print('\n ')

        # Create a dictionary with existing arrays
        all_arrays = {key: self.data[key] for key in self.data.files}
        # Add your new array
        all_arrays["grade_estimate"] = grade
        all_arrays["human_review"] = human_review
        # # Save everything to a new .npz file:
        if broken_flag:
            graded_npz_filename = self.npz_filename + '_RERUN_THIS_BENCHMARK.npz'
        elif np.sum(human_review) == 0:
            graded_npz_filename = self.npz_filename + '_final_grade.npz'
        else:
            graded_npz_filename = self.npz_filename + '_provisional_grade.npz'
        save_path = os.path.join(self.graded_benchmark_by_model_path, graded_npz_filename)
        np.savez(save_path, **all_arrays)

    #def quantitative_grader(self, solution, response):

    def deterministic_grader_ABC(self, solution, response, choices=['A', 'B', 'C']):
        """
        Checks if the correct multiple-choice answer is found in the response.

        Parameters:
        - solution (str): The correct answer ('A', 'B', or 'C').
        - response (str): The text response to be scanned.
        - choices (list): The possible choices (default: ['A', 'B', 'C']).

        Returns:
        - bool: True if the correct answer is found in the response, False otherwise.
        """

        # Ensure solution is a valid choice
        if solution not in choices:
            raise ValueError(f"Invalid solution '{solution}', must be one of {choices}")

        # Normalize whitespace and get the last line of the response
        lines = response.strip().split("\n")
        last_line = lines[-1].strip() if lines else ""

        # Regular expressions to detect explicit answer declarations
        explicit_answer_patterns = [
            rf"The answer would be\s*['\"]{solution}['\"]",
            rf"^\s*\*\*{solution}\*\*\s*$",
            rf"\\\[\s*\\boxed\{{\s*{solution}\s*\}}\s*\\\]",
            rf"^\s*{solution}\s*$",
            # Simple sentence-style answer declarations
            rf"\bThe final answer is:?\s*{solution}\b",
            rf"\bThe correct answer is:?\s*{solution}\b",
            rf"\bThe answer is:?\s*{solution}\b",
            rf"\bAnswer:\s*{solution}\b",
            rf"^\s*{solution}\s*\(.*?\)\s*$",
            # Bolded answer declarations
            rf"\*\*?Final answer:?\*\*?\s*\**{solution}\**", # e.g. **Final answer:** **B**
            rf"\*\*Answer:\s*{solution}\*\*?",  # **Answer: A or **Answer: A**
            rf"\*\*{solution}\s*\(.*?\)\*\*",  # **C (uncertain)**
            rf"\*\*Answer:\s*{solution}\s*\(.*?\)\.\*\*"
            # Bolded answers on a new line after bolded intro
            rf"\*\*\s*\n+\s*\*\*{solution}\.",
            # **\n\n**A.
            rf"The answer is:\*\*\s*\n+\s*\*\*{solution}\.",
            # The answer is:**\n\n**A.
            rf"\bFinal Answer\s*\n+\s*\*\*{solution}\b",
            # Final Answer\n\n**C
            rf"\bFinal Answer\s*\n+\s*\*\*{solution}\*\*",
            # Final Answer\n\n**A**
            rf"\*\*Final answer:\*\*\s*\n\s*{solution}\b",
            # **Final answer:** \nC
            rf"\n+\s*\*\*Final Answer:\s*{solution}\*\*",
            # \n\n**Final Answer: A**
            rf"\*\*Final answer:\*\*\s*\n\s*\*\*{solution}\*\*",
            # **Final answer:**  \n**B**
            rf"\*\*Final Answer:\*\*\s*\n\s*\*\*{solution}\*\*",
            # **Final Answer:**  \n**B**
            rf"\*\*Answer:\s*{solution}\*\*",
            # **Answer: C**
            rf"\banswer:\s*\n+\s*\*\*{solution}\b",
            # answer:\n\n**A
            rf"answer is:\*\*\s*\n+\s*\*\*{solution}\b",
            # answer is:**\n\n**C
            rf"\n+\s*\*\*Answer:\s*{solution}\b",
            # \n\n**Answer: A
            rf"correct answer is:\*\*\s*\n+\s*\*\*{solution}\b",
            # correct answer is:**\n\n**C
            rf"\*\*\s*\n\s*\*\*{solution}\*\*", 
            # **  \n**A**    
            # Bold answer declarations with explanation in parentheses
            rf"\*\*\s*\n+\s*\*\*Answer:\s*{solution}.*\*\*", 
            # **\n\n**Answer: A (yes/no/...)
            rf"^\s*{solution}\s—\s",
            rf"we can conclude:\s*{solution}\b",
            rf"the most appropriate answer under these conditions would be ['\"]{solution}['\"]",
            rf"lean\s+towards\s+['\"]{solution}['\"]",
            rf"tentatively\s+answer\s+['\"]{solution}['\"]",
            rf"\s*{solution}\s+\(No\)\.\s*",
            rf"\s*{solution}\s+\(Yes\)\.\s*",
            rf"\s*{solution}\s+\(Uncertain\)\.\s*",
            rf"{solution}\s*-\s*No\.\s*",
            rf"{solution}\s*-\s*Yes\.\s*",
            rf"{solution}\s*-\s*Uncertain\.\s*",
            rf"is:\s*{solution}\s*-\s*No",
            rf"is:\s*{solution}\s*-\s*Yes", 
            rf"is:\s*{solution}\s*-\s*Uncertain"
        ]

        # Check if the last line explicitly declares the answer
        for pattern in explicit_answer_patterns:
            if re.search(pattern, last_line, re.IGNORECASE):
                return True
        # As soon as re.search(...) finds a match, the function
        # returns True immediately

        # General pattern to check if the answer appears any where in the response
        pattern = rf"\b{solution}\b|the answer is {solution}"  
        # Perform case-insensitive regex search
        
        match = re.search(r'Answer:\s*(?:\$?\\?boxed\{)?["\'\$\\\s]*([A-Ca-c])["\'\}\$\\\s]*', response)
        is_correct = True # Answer is assumed to be true unless otherwise specified.
        if match:
            extracted = str(match.group(1)).strip().upper()
            expected = f'{solution}'.strip().upper()

            # Optional: strip any non-A/B/C just in case
            extracted = re.sub(r'[^A-C]', '', extracted)
            is_correct = extracted == expected
            return is_correct
        else:
            extracted = "INVALID"
            is_correct = False
            return is_correct