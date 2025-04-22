import re

class Grader():

    def __init__(self, benchmark, responses):

        exam_name = benchmark['name'][0]

        n = len(benchmark['question'])
        solutions = benchmark["solution"]
        grade = np.zeros([n])
        for i in range(0,n):
            if exam_name.startswith("MediatedCausality"):
                correct = self.grade_string_multiple_choice(
                    solutions[i],
                    responses[i],
                    choices=['A', 'B', 'C']
                )
            else:
                correct = self.grade_string_exactly(
                    solutions[i],
                    responses[i]
                )
            if correct:
                grade[i] = 1.0
            else:
                grade[i] = 0.0
        
        self.grade = grade
        self.responses = responses

    def get_grades(self):
        return self.grade
    
    def get_responses(self):
        return self.responses

    def assertExactlyEqualStrings(self, first, second):
        if first == second:
            return True
        else:
            return False

    def assertAlmostEqualNumbers(self, first, second, places=1):
        """
        False if the two objects are unequal as determined by their rounded difference
        to the given number of decimal places (default 7) and compare them as equal.
        """
        msg=True
        difference = round(abs(first - second), places)
        # If the difference is not zero, the test fails
        if difference != 0:
            msg=False
        return msg

    def grade_numerical(self,solution,response):
        return self.assertAlmostEqual(solution,response, places=10) #  (default tolerance is places=7 decimal places)

    def grade_string_exactly(self,solution,response):
        return self.assertExactlyEqualStrings(solution,response)

    def grade_string_multiple_choice(self,solution,response,choices):
        return self.assertExactlyEqualStrings(solution,response)

    def grade_string_multiple_choice(self, solution, response, choices=['A', 'B', 'C']):
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
            rf"\bThe final answer is:?\s*{solution}\b",
            rf"\bThe correct answer is:?\s*{solution}\b",
            rf"\bThe answer is:?\s*{solution}\b",
            rf"\bAnswer:\s*{solution}\b",
            rf"\*\*?Final answer:?\*\*?\s*\**{solution}\**",
            rf"\*\*?Answer:?\s*{solution}",
            rf"The answer is:\*\*\s*\n+\s*\*\*{solution}\.", 
            rf"\bFinal Answer\s*\n+\s*\*\*{solution}\b",
            rf"\bFinal Answer\s*\n+\s*\*\*{solution}\*\*",
            rf"\*\*Final answer:\*\*\s*\n\s*{solution}\b",
            rf"\n+\s*\*\*Final Answer:\s*{solution}\*\*",
            rf"\*\*Final answer:\*\*\s*\n\s*\*\*{solution}\*\*",
            rf"\*\*Answer:\s*{solution}\*\*",
            rf"\banswer:\s*\n+\s*\*\*{solution}\b",
            rf"answer is:\*\*\s*\n+\s*\*\*{solution}\b",
            rf"\n+\s*\*\*Answer:\s*{solution}\b",
            rf"correct answer is:\*\*\s*\n+\s*\*\*{solution}\b", 
        ]

        # Check if the last line explicitly declares the answer
        for pattern in explicit_answer_patterns:
            if re.search(pattern, last_line, re.IGNORECASE):
                return True
        # As soon as re.search(...) finds a match, the function
        # returns True immediately

        # General pattern to check if the answer appears anywhere in the response
        general_pattern = rf"\b{solution}\b|the answer is {solution}"

        # Perform case-insensitive regex search throughout the response
        match = re.search(general_pattern, response, re.IGNORECASE)

        return bool(match)  # Return True if match is found, else False


#===============================================================================

# # Example Usage
# solution = 'A'
# response1 = "The answer is A."  # ✅ Should return True
# response2 = "I think A."  # ✅ Should return True
# response3 = "A dog is friendly."  # ❌ Should return False
# response4 = "The answer is B."  # ❌ Should return False
#
# print(grade_string_multiple_choice(solution, response1))  # True
# print(grade_string_multiple_choice(solution, response2))  # True
# print(grade_string_multiple_choice(solution, response3))  # False
# print(grade_string_multiple_choice(solution, response4))  # False
