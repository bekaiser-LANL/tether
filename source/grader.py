import torch
from PIL import Image
import csv
import os
from sentence_transformers import SentenceTransformer, util
import re

class grader():

    def assertEqualImageDescriptions(self, solution,response):
        # Load CLIP for text embeddings
        model = SentenceTransformer("/lustre/scratch5/dmperez/LLMs/local_all_MiniLM_L6_v2")
        self.path = "/lustre/scratch5/dmperez/Tether/source/benchmarks/equations/"

        reference_file = "/lustre/scratch5/dmperez/Tether/source/benchmarks/equation_labels.txt"
        reference_dict = {}
        with open(reference_file, "r") as f:
            for line in f:
                if "\t\t" in line:
                    parts = line.strip().split("\t\t")
                    if len(parts) >= 2:
                        filename = parts[0].strip()
                        keywords = [kw.strip() for kw in parts[1].split(",")]  # assuming comma-separated labels
                       # print(f"✅ File: {filename}")
                       # print(f"✅ Keywords for this row: {keywords}")
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

    def assertExactlyEqualStrings(self, first, second):
        if first == second:
            return True
        else:
            return False

    def assertAlmostEqualNumbers(self,first, second, places=1):
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
        return self.assertEqualImageDescriptions(solution,response)

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

        
        # Regular expression pattern to match the exact choice
        pattern = rf"\b{solution}\b|the answer is {solution}"  # \b ensures word boundaries

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
