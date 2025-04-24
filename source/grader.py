""" Classes and functions for grading LLM responses """
import os
import re

# NEEDS TO BE LINTED
# HARD-CODED PATHS MUST BE REMOVED

class Grader():

    def __init__(self, benchmark, responses):

        # NEEDS TO BE MOVED OUTSIDE OF THIS SCRIPT (export / .bashrc, .zshrc)
        self.path = "/lustre/scratch5/dmperez/Tether/source/benchmarks/equations/"

        exam_name = benchmark['name'][0]

        n = len(benchmark['question'])
        solutions = benchmark["solution"]
        grade = np.zeros([n])
        for i in range(0,n):
            if exam_name.startswith("MediatedCausality") or exam_name.endswith("Inequality"):
                correct = self.grade_string_multiple_choice(
                    solutions[i],
                    responses[i],
                    choices=['A', 'B', 'C']
                )
            elif exam_name == 'equations':
                correct = self.grader.grade_images(
                    self.solutions[index],
                    response
                )
                continue
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

    def assert_equal_image_descriptions(self, solution,response):
        # Load CLIP for text embeddings
        model = SentenceTransformer("/lustre/scratch5/dmperez/LLMs/local_all_MiniLM_L6_v2")
        reference_file = "/lustre/scratch5/dmperez/Tether/source/benchmarks/equation_labels.txt"
        reference_dict = {}
        with open(reference_file, "r") as f:
            for line in f:
                if "\t\t" in line:
                    parts = line.strip().split("\t\t")
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        keywords = [kw.strip() for kw in parts[1].split(",")]  # assuming comma-separated labels
                       # print(f" File: {filename}")
                       # print(f" Keywords for this row: {keywords}")
                    # Convert keywords to a readable sentence
                    if len(keywords) == 1:
                        caption = f"A photo of {keywords[0]}."
                    elif len(keywords) == 2:
                        caption = f"A photo of {keywords[0]} and {keywords[1]}."
                    else:
                        caption = f"A photo of {', '.join(keywords[:-1])}, and {keywords[-1]}."
                    reference_dict[filename] = caption

        # Get all image files in the directory (supports PNG, JPG, JPEG)
        image_files = [f for f in os.listdir(self.path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        #results = []
        for image_file in image_files:
            print(image_file)
            image_filename = os.path.basename(self.path+image_file)  # Extract "blueberries.jpg"
            ref_caption = reference_dict.get(image_filename, "")

            if not ref_caption:
                print(f"⚠️ No reference found for {image_filename}. Skipping.")
                continue

            # Compute embeddings 
            embeddings = model.encode([response, ref_caption], convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()


            print(f"{filename}:")
            print(f"  Generated: {response}")
            print(f"  Reference: {ref_caption}")
            print(f"  Cosine similarity: {similarity:.4f}\n")

            return similarity

    def assert_exactly_equal_strings(self, first, second):
        if first == second:
            return True
        else:
            return False

    def assert_almost_equal_numbers(self, first, second, places=1):
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

    def grade_images(self,solution,response):
        return self.assert_equal_image_descriptions(solution,response)

    def grade_numerical(self,solution,response):
        return self.assert_almost_equal_numbers(solution,response, places=10) #  (default tolerance is places=7 decimal places)

    def grade_string_exactly(self,solution,response):
        return self.assert_exactly_equal_strings(solution,response)

    def grade_string_multiple_choice(self,solution,response):
        return self.assert_exactly_equal_strings(solution,response)

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
            # Simple sentence-style answer declarations
            rf"\bThe final answer is:?\s*{solution}\b",
            rf"\bThe correct answer is:?\s*{solution}\b",
            rf"\bThe answer is:?\s*{solution}\b",
            rf"\bAnswer:\s*{solution}\b",

            # Bolded answer declarations
            rf"\*\*?Final answer:?\*\*?\s*\**{solution}\**", # e.g. **Final answer:** **B**
            rf"\*\*?Answer:?\s*{solution}", # e.g. **Answer: A

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
            return extracted, is_correct

        return is_correct #bool(match)  # Return True if match is found, else False


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
