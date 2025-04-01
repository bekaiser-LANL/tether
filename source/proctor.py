from .grader import grader
from .benchmark import benchmark
import numpy as np
import subprocess
import requests
import os

class proctor():

    def __init__(self, settings, exam_name, checkpoints=np.nan, restart=np.nan):
        self.path       = settings['path'] + '/benchmarks/completed/' # path to benchmark reports
        self.reuse      = settings['path'] + '/benchmarks/saved/' # path to saved benchmark
        self.figures    = settings['path'] + '/benchmarks/figures/'
        self.model      = settings['model']
        self.n_problems = settings['n_problems']
        self.grader     = grader()
        self.generate   = settings['generate']
        self.exam_idx   = settings['exam_idx']
        self.record_txt = settings['record_txt']   
        self.temperature  = settings['temperature'] # for OpenAI non-reasoning models only
        self.reasoning_effort = settings['reasoning_effort'] # for OpenAI reasoning models only
        self.n_numbers = settings['n_numbers'] # for standardDeviation benchmark only 
        self.checkpoints = checkpoints # Checkpoint frequency if an integer, no checkpoint .npz output if a NaN
        self.restart = restart # Restart question number if an integer, start at question 1 if a NaN

        self.create_missing_directory(self.path)
        self.create_missing_directory(self.reuse)
        self.create_missing_directory(self.figures)

        # **********************************************************************
        # 1) generate new benchmark ********************************************

        if self.generate:

            if exam_name == 'significantFigures':

                from .benchmarks.significantFigures import significantFigures
                self.exam = significantFigures(n_problems=self.n_problems)

            elif exam_name == 'standardDeviation':

                from .benchmarks.standardDeviation import standardDeviation
                self.exam = standardDeviation(n_numbers=self.n_numbers, n_problems=self.n_problems)

            elif exam_name == 'mediatedCausalitySmoking' or exam_name == 'mediatedCausalitySmokingWithMethod':

                from .benchmarks.mediatedCausality import mediatedCausality
                x_name = 'smoke'
                z_name = 'have tar deposits in lungs'
                y_name = 'have lung cancer'
                x_name_verb = 'smoking'
                y_name_noun = 'lung cancer'
                name_list = [x_name,z_name,y_name,x_name_verb,y_name_noun]
                plot_path = self.reuse + exam_name + '_figures/'
                self.exam = mediatedCausality(plot_path,exam_name,name_list=name_list,plot_flag=True,n_problems=self.n_problems)

        else: # use a saved benchmark

            # # read saved .txt files
            # if self.is_integer(self.exam_idx):
            #     read_path = self.reuse + exam_name + '_' + str(self.exam_idx) + '.txt'
            # else:
            #     read_path = self.reuse + exam_name + '.txt'   
            # from .benchmarks.readSavedBenchmarkTxt import readSavedBenchmarkTxt
            # self.exam = readSavedBenchmarkTxt(read_path,exam_name)

            # read saved .npz files
            read_path = self.reuse + exam_name + '_' + str(self.exam_idx) + '.npz'
            from .benchmarks.readSavedBenchmarkNpz import readSavedBenchmarkNpz
            self.exam = readSavedBenchmarkNpz(read_path)


        # For grading and saving:
        self.metadata  = self.exam.get_metadata()
        self.questions = self.exam.get_questions()
        self.solutions = self.exam.get_solutions()
        grade = np.zeros([self.n_problems])*np.nan
        length_str = '\n Number of questions: %i' %self.n_problems
        model_str = '\n Model: ' + self.model
        exam_str = '\n Exam: ' + exam_name
        temp_str = '\n Temperature: ' + str(self.temperature)
        effort_str = '\n Reasoning effort: ' + self.reasoning_effort
        self.benchmark = benchmark(self.path,self.model,self.exam)

        report = {
            "exam_name": exam_name,
            "model_str": model_str,
            "exam_str": exam_str,
            "length_str": length_str,
            "temp_str": temp_str,    
            "effort_str": effort_str,                       
            "questions": self.questions,
            "solutions": self.solutions,      
            "exam_idx": self.exam_idx,
            "reuse": self.reuse,
            "checkpoints": self.checkpoints,
            "metadata": self.metadata          
        }

        # save an npz file of this exam:
        if self.generate: 
            self.benchmark.save_blank_exam_npz(report) 
        if self.record_txt: # not recommended
            self.benchmark.save_blank_exam_txt(report)     

        # **********************************************************************
        # 2) assess the model using the benchmark ******************************

        if self.model == "llama3.2" or self.model == "llama3":

            server = subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            grade_ = []
            correct_ = []      
            response_ = []     
            counter = []
               
            # restart question index:
            if np.isnan(self.restart):
                i0 = 0
            else:
                i0 = int(self.restart)   
            for i in range(i0,self.n_problems): # length of exam 

                # try:

                # API endpoint
                url = "http://localhost:11434/api/generate"

                # write the name of the LLM at the top of the exam

                # Request payload
                payload = {
                    "model": self.model,
                    "prompt": self.questions[i], # ! call Tether for prompt
                    "stream": False
                }

                # Send the request to the API
                llm_response = requests.post(url, json=payload)

                if llm_response.status_code == 200:

                    # This is the standard HTTP status code for a successful request.
                    # Successful response from the Ollama API

                    # grade:
                    response = llm_response.json().get("response")
                    if exam_name == 'mediatedCausalitySmoking' or exam_name == 'mediatedCausalitySmokingWithMethod':
                        correct = self.grader.grade_string_multiple_choice(self.solutions[i],response,choices=['A', 'B', 'C'])
                    else:
                        correct = self.grader.grade_string_exactly(self.solutions[i],response)
                    if correct:
                        grade[i] = 1.0
                    else:
                        grade[i] = 0.0

                else:
                    response  = 'Error! Internet connection to Ollama broken.' #' Status_code: ' + response.status_code + response.text
                    correct   = 'N/A'

                # Save the LLM performance on the benchmark (the "report card"):
                report['grade'] = grade
                report['correct'] = correct       
                report['question_idx'] = i # needed for txt report only
                report['response'] = response  
                counter = np.append(counter,i)
                report['counter'] = counter                  
                self.print_questions_to_terminal(report)                        
                self.benchmark.write_txt_report(report) 

                grade_ = np.append(grade_,grade)
                correct_ = np.append(correct_,correct)
                response_ = np.append(response_,response) 

                if not np.isnan(self.checkpoints):
                    if self.max_multiple(int(i+1),self.checkpoints) == int(i+1):
                        report['checkpoints'] = int(self.max_multiple(int(i+1),self.checkpoints))
                        self.benchmark.write_npz_report(report, grade_, correct_, response_) 


                # finally:
                #     # Stop the Ollama server
                #     server.terminate()
                #     server.wait()

            self.benchmark.write_npz_report(report, grade_, correct_, response_) 

        elif self.model == "gpt-4.5-preview" or self.model == "gpt-4o" or self.model == "o3-mini" or self.model == "o1":

            # you need your own OpenAI API key in your .zshrc or .bashrc for this to work:

            from openai import OpenAI
            client = OpenAI()
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")     
            self.client = OpenAI(api_key=OPENAI_API_KEY)

            grade_ = []
            correct_ = []      
            response_ = []    
            counter = []

            print(' Verify these are all the same: = ',len(self.questions),len(self.solutions),self.n_problems)

            # restart question index:
            if np.isnan(self.restart):
                i0 = 0
            else:
                i0 = int(self.restart)   
            for i in range(i0,self.n_problems): # length of exam 

                # try:

                # Send the request to the API
                response = self.ask_openai(self.questions[i],self.model)
        
                # grade:
                if exam_name == 'mediatedCausalitySmoking' or exam_name == 'mediatedCausalitySmokingWithMethod':
                    correct = self.grader.grade_string_multiple_choice(self.solutions[i],response,choices=['A', 'B', 'C'])
                else:
                    correct = self.grader.grade_string_exactly(self.solutions[i],response)
                if correct:
                    grade[i] = 1.0
                else:
                    grade[i] = 0.0

                report['grade'] = grade
                report['correct'] = correct       
                report['question_idx'] = i 
                report['response'] = response  
                counter = np.append(counter,i)
                report['counter'] = counter                      
                self.print_questions_to_terminal(report)                                        
                self.benchmark.write_txt_report(report) 
                
                grade_ = np.append(grade_,grade)
                correct_ = np.append(correct_,correct)
                response_ = np.append(response_,response) 

                if not np.isnan(self.checkpoints):
                    if self.max_multiple(int(i+1),self.checkpoints) == int(i+1):
                        report['checkpoints'] = int(self.max_multiple(int(i+1),self.checkpoints))
                        self.benchmark.write_npz_report(report, grade_, correct_, response_) 

                # except Exception as e:
                #     print('\n An error occured for model ',self.model)
                #     return f"Error: {e}"  

            self.benchmark.write_npz_report(report, grade_, correct_, response_)  

    def max_multiple(self, a: int, b: int) -> int:
        """
        Returns the maximum multiple of b that is less than or equal to a.

        Parameters:
        - a (int): The upper limit.
        - b (int): The base multiple.

        Returns:
        - int: The highest multiple of b that does not exceed a.
        """
        return (a // b) * b  # Finds the largest multiple of b within a

    def create_missing_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def is_integer(self, value):
        return isinstance(value, int)

    def print_questions_to_terminal(self, report):
        print('\n ', report['question_idx'])
        print(' question: ',report['questions'][report['question_idx']])
        print(' response: ',report['response'])   
        print(' correct: ',report['solutions'][report['question_idx']])     

    #def verify      

    def ask_openai(self, question, model_choice, reasoning_effort='medium'):

        if model_choice == 'gpt-4o' or model_choice == 'gpt-4.5-preview':
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
            except Exception as e:
                return f"Error: {e}"
        elif model_choice == 'o3-mini' or model_choice == 'o1':
            try:
                response = self.client.chat.completions.create(model=model_choice,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                reasoning_effort=self.reasoning_effort # Options: 'low', 'medium', 'high')
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                return f"Error: {e}"            
