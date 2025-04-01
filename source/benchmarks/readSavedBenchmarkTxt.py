import numpy as np
import math as ma
import random
import re
import os

class readSavedBenchmarkTxt():
    def __init__(self, read_path, exam_name):
        self.read_path = read_path
        self.exam_name = exam_name

        print('\n Reading saved benchmark')

        extracted_data = []
        current_question = None
        current_solution = None
        with open(self.read_path, "r") as file:
            for line in file:
                # Match text after "Problem: "
                problem_match = re.search(r"Problem:\s*(.+)", line)
                #problem_match = re.search(r"Problem:\s*(.*)", line)
                if problem_match:
                    current_question = problem_match.group(1).strip()

                # Match text after "Solution: "
                if exam_name == 'mediatedCausalitySmoking' or exam_name == 'mediatedCausalitySmokingWithMethod':
                    solution_match = re.search(r"Solution:\s*([A-C])", line)
                    if solution_match:
                        current_solution = solution_match.group(1).strip()
                elif exam_name == 'standardDeviation':
                    solution_match = re.search(r"Solution:\s*([\d\.]+)", line)
                    if solution_match:
                        current_solution = solution_match.group(1).strip()
                elif exam_name == 'significantFigures':
                    #solution_match = re.search(r"Solution:\s*([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)", line)
                    solution_match = re.search(r"Solution:\s*(.+)", line)
                    if solution_match:
                        current_solution = solution_match.group(1).strip()

                # Store if both question and solution are found
                if current_question and current_solution:
                    extracted_data.append((current_question, current_solution))
                    current_question = None  # Reset for the next problem
                    current_solution = None  # Reset for the next solution

        self.questions = [] #np.empty((), dtype=object)
        self.solutions = [] #np.empty((), dtype=object)
        for i, (question, solution) in enumerate(extracted_data):
            self.questions = np.append(self.questions,question)
            self.solutions = np.append(self.solutions,solution)
        self.n_problems = len(self.questions)

        self.metadata = {
            "Name": exam_name,
            "n_problems": len(self.questions)
        }

        extractor = metadataExtractor(self.read_path)
        if exam_name == 'mediatedCausalitySmoking' or exam_name == 'mediatedCausalitySmokingWithMethod':
            # top of file metadata:
            tmp = extractor.extract_causal_exam_statistics()
            A_perc = tmp[0]; B_perc = tmp[1]; C_perc = tmp[2];
            n_problems = tmp[3]
            if n_problems != self.metadata['n_problems']:
                print('\n WARNING: Error in metadata read:')
                print(' n_problems: %i neq %i ' %(n_problems,self.metadata['n_problems']))
            self.metadata['A_count'] = int(A_perc*n_problems/100.)
            self.metadata['B_count'] = int(B_perc*n_problems/100.)
            self.metadata['C_count'] = int(C_perc*n_problems/100.)
            sum = int(self.metadata['A_count']+self.metadata['B_count']+self.metadata['C_count'])
            if sum != n_problems:
                print('\n WARNING: A,B,C answer counts do not add up to n_problems')
                print(' n_problems: %i neq %i ' %(n_problems,sum))
            tmp = []

            # problem metadata:
            tmp = extractor.extract_causal_probability_values()
            P_Y1doX1 = tmp['P(Y=1|do(X=1))']; P_Y1doX0 = tmp['P(Y=1|do(X=0))']
            print("Length of P_Y1doX1:", len(P_Y1doX1))
            print("Length of P_Y1doX0:", len(P_Y1doX0))

            self.metadata['P_Y1doX1'] = [p[0] for p in P_Y1doX1]
            self.metadata['P_Y1doX1_CI'] = [p[1] for p in P_Y1doX1]
            self.metadata['P_Y1doX0'] = [p[0] for p in P_Y1doX0]
            self.metadata['P_Y1doX0_CI'] = [p[1] for p in P_Y1doX0]
            tmp = []

            print('\n len(self.metadata[P_Y1doX1]) = ',len(self.metadata['P_Y1doX1']) )
            print(' n_problems = ',n_problems)
            print(' self.metadata[n_problems] = ',self.metadata['n_problems'])           

    def get_questions(self): # all tests need this
        return self.questions

    def get_solutions(self): # all tests need this
        return self.solutions

    def get_metadata(self): # all tests need this
        return self.metadata

class metadataExtractor():
    def __init__(self, file_path):
        self.file_path = file_path

    def extract_causal_exam_statistics(self):
        """ Extracts the four numbers from the statistics section. """
        numbers = []
        with open(self.file_path, "r") as file:
            inside_section = False  # Track when inside the target section

            for line in file:
                # Detect the start and end of the statistics section
                if "========================================" in line:
                    if inside_section:
                        break  # End extraction when the second occurrence is found
                    inside_section = True
                    continue  # Skip the line itself

                if inside_section:
                    # Extract numbers from the current line
                    found_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    numbers.extend(found_numbers)

        return list(map(float, numbers))  # Convert all numbers to float

    def extract_causal_probability_values(self):
        """Extracts all values from 'P(Y=1|do(X=1))' and 'P(Y=1|do(X=0))' equations."""
        probability_values = {"P(Y=1|do(X=1))": [], "P(Y=1|do(X=0))": []}

        expected_count = 1800  # We expect 1800 values
        count_x1 = 0
        count_x0 = 0

        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                count_x1 += line.count("P(Y=1|do(X=1))")
                count_x0 += line.count("P(Y=1|do(X=0))")

        print(f"🔍 Found {count_x1} occurrences of P(Y=1|do(X=1)) (Expected: {expected_count})")
        print(f"🔍 Found {count_x0} occurrences of P(Y=1|do(X=0)) (Expected: {expected_count})")

        # ✅ Use findall() to capture multiple matches per line
        pattern_x1 = re.compile(r"P\(Y=1\|do\(X=1\)\)\s*=\s*([\d.eE+-]+)\s*(?:\+/-|±)\s*([\d.eE+-]+)")
        pattern_x0 = re.compile(r"P\(Y=1\|do\(X=0\)\)\s*=\s*([\d.eE+-]+)\s*(?:\+/-|±)\s*([\d.eE+-]+)")

        with open(self.file_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                line = re.sub(r"\s+", " ", line)  # Normalize multiple spaces
                
                # Extract all matches in the line
                matches_x1 = pattern_x1.findall(line)
                matches_x0 = pattern_x0.findall(line)

                # Store matches properly
                for match in matches_x1:
                    probability_values["P(Y=1|do(X=1))"].append((float(match[0]), float(match[1])))

                for match in matches_x0:
                    probability_values["P(Y=1|do(X=0))"].append((float(match[0]), float(match[1])))

        # ✅ Final check to confirm all values are captured
        len_x1 = len(probability_values["P(Y=1|do(X=1))"])
        len_x0 = len(probability_values["P(Y=1|do(X=0))"])
        
        if len_x1 != expected_count or len_x0 != expected_count:
            print(f"⚠️ WARNING: Expected {expected_count}, but found {len_x1} P(Y=1|do(X=1))")
            print(f"⚠️ WARNING: Expected {expected_count}, but found {len_x0} P(Y=1|do(X=0))")

        return probability_values




    # def extract_causal_probability_values(self):
    #     """ Extracts all values from 'P(Y=1|do(X=1))' and 'P(Y=1|do(X=0))' equations. """
    #     probability_values = {"P(Y=1|do(X=1))": [], "P(Y=1|do(X=0))": []}

    #     with open(self.file_path, "r") as file:

    #         for line in file:
    #             line = line.strip()  # Ensure clean input

    #             # Extract P(Y=1|do(X=1)) values
    #             match_x1 = re.search(r"P\(Y=1\|do\(X=1\)\)\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", line)
    #             if match_x1:
    #                 probability_values["P(Y=1|do(X=1))"].append((float(match_x1.group(1)), float(match_x1.group(2))))

    #             # Extract P(Y=1|do(X=0)) values
    #             match_x0 = re.search(r"P\(Y=1\|do\(X=0\)\)\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)", line)
    #             if match_x0:
    #                 probability_values["P(Y=1|do(X=0))"].append((float(match_x0.group(1)), float(match_x0.group(2))))

    #     return probability_values

    # def extract_causal_probability_values(self):
    #     """Extracts all values from 'P(Y=1|do(X=1))' and 'P(Y=1|do(X=0))' equations."""
    #     probability_values = {"P(Y=1|do(X=1))": [], "P(Y=1|do(X=0))": []}

    #     with open(self.file_path, "r", encoding="utf-8") as file:
    #         for line in file:
    #             line = line.strip()  # Ensure clean input
                
    #             # Extract P(Y=1|do(X=1)) values
    #             match_x1 = re.search(r"P\(Y=1\|do\(X=1\)\)\s*=\s*([\d.eE+-]+)\s*(?:\+/-|±)\s*([\d.eE+-]+)", line)
    #             if match_x1:
    #                 probability_values["P(Y=1|do(X=1))"].append((float(match_x1.group(1)), float(match_x1.group(2))))

    #             # Extract P(Y=1|do(X=0)) values
    #             match_x0 = re.search(r"P\(Y=1\|do\(X=0\)\)\s*=\s*([\d.eE+-]+)\s*(?:\+/-|±)\s*([\d.eE+-]+)", line)
    #             if match_x0:
    #                 probability_values["P(Y=1|do(X=0))"].append((float(match_x0.group(1)), float(match_x0.group(2))))

    #     return probability_values

    # def extract_causal_probability_values(self):
    #     """Extracts all values from 'P(Y=1|do(X=1))' and 'P(Y=1|do(X=0))' equations."""
    #     probability_values = {"P(Y=1|do(X=1))": [], "P(Y=1|do(X=0))": []}
    #     missed_lines = []  # Tracks lines where regex failed

    #     with open(self.file_path, "r", encoding="utf-8") as file:
    #         for line_number, line in enumerate(file, start=1):
    #             line = line.strip()

    #             # Skip Irrelevant Lines
    #             if not line.startswith("P(Y=1|do(X="):  
    #                 continue  # Ignore headers, metadata, and random text
                
    #             # Normalize spaces and special characters
    #             line = re.sub(r"[^\x00-\x7F]+", " ", line)  # Remove non-ASCII characters
    #             line = re.sub(r"\s+", " ", line)  # Normalize multiple spaces

    #             # Define regex patterns with broader flexibility
    #             patterns = [
    #                 r"P\(Y=1\|do\(X=1\)\)\s*=\s*([\d.,eE+-]+)\s*(?:\+/-|±)\s*([\d.,eE+-]+)",
    #                 r"P\(Y=1\|do\(X=0\)\)\s*=\s*([\d.,eE+-]+)\s*(?:\+/-|±)\s*([\d.,eE+-]+)"
    #             ]

    #             # Extract P(Y=1|do(X=1))
    #             match_x1 = re.search(patterns[0], line)
    #             if match_x1:
    #                 probability_values["P(Y=1|do(X=1))"].append((float(match_x1.group(1).replace(",", "")), 
    #                                                             float(match_x1.group(2).replace(",", ""))))
    #             else:
    #                 missed_lines.append((line_number, "P(Y=1|do(X=1))", line))

    #             # Extract P(Y=1|do(X=0))
    #             match_x0 = re.search(patterns[1], line)
    #             if match_x0:
    #                 probability_values["P(Y=1|do(X=0))"].append((float(match_x0.group(1).replace(",", "")), 
    #                                                             float(match_x0.group(2).replace(",", ""))))
    #             else:
    #                 missed_lines.append((line_number, "P(Y=1|do(X=0))", line))

    #     # Print only if there are missing lines
    #     if len(missed_lines) > 0 and len(missed_lines) < 20:  # Avoid printing thousands of lines
    #         print(f"\n⚠️ WARNING: {len(missed_lines)} lines failed to match regex!")
    #         for line_info in missed_lines[:5]:  # Show first few problematic lines
    #             print(f"Line {line_info[0]} ({line_info[1]} issue): {line_info[2]}")

    #     return probability_values





#===============================================================================
# Tests

# # Example usage:
# file_path = "example.txt"  # Replace with your actual file path
# extractor = NumberExtractor(file_path)
#
# # Extract four main numbers
# statistics_numbers = extractor.extract_statistics_numbers()
# print("Extracted Statistics Numbers:", statistics_numbers)
#
# # Extract probability values
# probability_values = extractor.extract_probability_values()
# print("\nExtracted Probability Values:")
# for key, values in probability_values.items():
#     print(f"{key}: {values}")
