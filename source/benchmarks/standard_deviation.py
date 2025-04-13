import numpy as np
import math as ma

class StandardDeviation():

    def __init__(self, range=[-100,100], n_numbers = 20, decimal_places=4, n_problems=100):
        self.n_problems = n_problems # all tests need this
        self.decimal_places = decimal_places
        self.range = range
        self.n_numbers = n_numbers
        self.metadata = {
            "Name": 'standardDeviation',
            "n_problems": self.n_problems            
        }
        self.make_problems() # all tests need this

    def make_problems(self): # all tests need this
        self.questions = [] # all tests need this
        self.solutions = [] # all tests need this
        for i in range(0,self.n_problems): # all tests need this

            random_numbers, q_str = self.generate_question()
            self.questions = np.append(self.questions,q_str)

            ans_str = '{:.{}f}'.format(np.round(np.std(random_numbers),self.decimal_places), int(self.decimal_places))
            self.solutions = np.append(self.solutions,ans_str)


    def generate_question(self):
        # Generate random numbers
        random_numbers = np.random.randint(self.range[0], self.range[1] + 1, self.n_numbers)

        # Convert the list of numbers to a space-separated string
        numbers_str = " ".join(map(str, random_numbers))

        # Construct the formatted question string
        q_str = f"What is the standard deviation of {numbers_str} to {self.decimal_places} decimal places? Only answer with the number."

        return random_numbers, q_str

    def print_problems(self): # all tests need this
        for i in range(0,self.n_problems):
            print('\n')
            print(self.questions[i])
            print(self.solutions[i])

    def get_questions(self): # all tests need this
        return self.questions

    def get_solutions(self): # all tests need this
        return self.solutions

    def get_metadata(self): # all tests need this
        return self.metadata
