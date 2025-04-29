""" Standard deviation calculation benchmark """
import numpy as np

class StandardDeviation():
    """ Biased (sample) standard deviation test """

    def __init__(self, **kwargs):
        """ Set up the exam """
    
        self.name = kwargs.get('exam_name')
        self.generate_flag = kwargs.get('generate_flag', True)
        self.verbose = kwargs.get('verbose', False)
        self.n_problems = kwargs.get('n_problems', 100)
        self.number_range = kwargs.get('number_range', [-100.,100.])
        self.n_numbers = kwargs.get('n_numbers', 10)

        if self.generate_flag:
            self.make_problems()

    def make_problems(self):
        """ Generate the problems """
        self.question = []
        self.biased_solution = []
        self.unbiased_solution = []
        self.example_idx = []
        for i in range(0,self.n_problems):
            random_numbers, q_str = self.generate_question()
            self.question = np.append(self.question,q_str)
            biased_ans = '{:.{}f}'.format(np.std(random_numbers), 10)
            self.biased_solution = np.append(self.biased_solution,biased_ans)
            unbiased_ans = '{:.{}f}'.format(np.std(random_numbers,ddof=1), 10)
            self.unbiased_solution = np.append(self.unbiased_solution,unbiased_ans)
            self.example_idx = np.append(self.example_idx,i)
            self.name = np.append(self.name,self.name[0])

    def generate_question(self):
        """ Generate random numbers """
        random_numbers = np.random.randint(
            self.number_range[0],
            self.number_range[1] + 1,
            self.n_numbers
        )
        # Convert the list of numbers to a space-separated string
        numbers_str = " ".join(map(str, random_numbers))
        # Construct the formatted question string
        q_str = f"What is the standard deviation of {numbers_str}?"
        return random_numbers, q_str

    def print_problems(self):
        for i in range(0,self.n_problems):
            print('\n')
            print(self.question[i])
            print(self.unbiased_solution[i])
            print(self.biased_solution[i])

    def get_questions(self):
        return self.question
