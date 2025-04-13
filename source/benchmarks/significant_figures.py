import numpy as np
import math as ma
import random

class SignificantFigures():

    def __init__(self, max_digits=10, n_problems=100):
        self.n_problems = n_problems # all tests need this
        self.max_digits = max_digits
        self.n_problems = n_problems
        self.metadata = {
            "Name": 'significantFigures',
            "n_problems": self.n_problems
        }
        self.make_problems() # all tests need this

    def make_problems(self): # all tests need this
        self.questions = [] # all tests need this
        self.solutions = [] # all tests need this
        for i in range(0,self.n_problems): # all tests need this
            number = self.make_random_number()
            digits = random.randint(1, self.max_digits)
            q_str = 'What is %30.30f in %i significant digits? Answer in scientific notation and use e (E notation). Only answer with the number.' %(number,digits)
            self.questions = np.append(self.questions,q_str) # all tests need this
            sci_str,dec_str = self.calculate_significant_digits(number, digits)
            self.solutions = np.append(self.solutions,sci_str) # all tests need this

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

    #===========================================================================

    def fifty_fifty(self):
        """
        Returns True or False with a 50/50 chance.
        """
        return random.choice([True, False])

    def make_random_number(self):

        # Generate random exponents
        low_exponent = 1   # Minimum exponent
        high_exponent = 14 # Maximum exponent
        num_samples = 2    # Number of random powers of 10 to generate

        # Generate random exponents and compute powers of 10
        random_exponents = np.random.randint(low_exponent, high_exponent + 1, size=num_samples)
        powers_of_10 = [int(10 ** exp) for exp in random_exponents]  # Ensure integer type

        above_decimal = np.float128(np.random.randint(low=1, high=powers_of_10[0],size=1))
        below_decimal = np.float128(1./(np.random.uniform(low=1.001, high=1000, size=None)))

        if self.fifty_fifty():
            random_number =  (np.float128(above_decimal + below_decimal))[0]
            # here add a check for .000000000 numbers, try again if it doesn't work. <----------------------------
            #if not self.all_decimal_places_zero(random_number):

        else:
            random_number = np.float128(below_decimal)

        return random_number

    def all_decimal_places_zero(self,num):
        """
        Checks if all the decimal places in a given number are zero.

        Parameters:
            num (float or int): The number to check.

        Returns:
            bool: True if all decimal places are zero, False otherwise.
        """
        if isinstance(num, int):
            # If the number is an integer, there are no decimal places
            return True

        # Convert the number to a string to check its fractional part
        fractional_part = str(num).split(".")[1] if "." in str(num) else ""

        # Check if all characters in the fractional part are zero
        return all(char == "0" for char in fractional_part)


    def ensure_period(self,s):
        """
        Checks if a string contains a period ('.'). If no period is present,
        appends a period to the end of the string.

        Parameters:
            s (str): The input string.

        Returns:
            str: The modified string with a period added if needed.
        """
        if '.' not in s:
            return s + '.'
        return s


    def scientific_to_decimal(self,sci_str):
        """
        Convert a scientific notation string to its exact decimal expression,
        preserving trailing zeros explicitly present in the input.

        Parameters:
            sci_str (str): A number in scientific notation as a string (e.g., "6.51500e+03").

        Returns:
            str: The exact decimal expression of the number (e.g., "6515.00").
        """
        try:
            # Split the input into mantissa and exponent parts
            if "e" in sci_str.lower():
                mantissa, exponent = sci_str.lower().split("e")
                exponent = int(exponent)  # Convert exponent to integer
            else:
                # If no scientific notation is present, return the input as it is
                return sci_str

            # Check if the mantissa has a fractional part
            if "." in mantissa:
                integer_part, fractional_part = mantissa.split(".")
            else:
                integer_part, fractional_part = mantissa, ""

            # Handle shifting the decimal point based on the exponent
            if exponent > 0:
                # Move the decimal point to the right
                fractional_part = fractional_part + "0" * exponent
                integer_part = integer_part + fractional_part[:exponent]
                fractional_part = fractional_part[exponent:]
            elif exponent < 0:
                # Move the decimal point to the left
                leading_zeros = "0" * (abs(exponent) - len(integer_part))
                fractional_part = leading_zeros + integer_part + fractional_part
                integer_part = "0"

            # Assemble the result
            if fractional_part:
                result = f"{integer_part}.{fractional_part}"
            else:
                result = integer_part

            # Preserve only explicitly present trailing zeros from the input mantissa
            if "." in mantissa:
                original_fractional_part = mantissa.split(".")[1]
                if original_fractional_part.endswith("0"):
                    required_length = len(original_fractional_part)
                    result = f"{integer_part}.{fractional_part}".ljust(len(integer_part) + 1 + required_length, "0")
                else:
                    result = f"{integer_part}.{fractional_part}".rstrip("0").rstrip(".")
            else:
                result = result.rstrip("0").rstrip(".")

            return result
        except Exception as e:
            return f"Error: {str(e)}"

    def calculate_significant_digits(self, number, digits):
        """
        Rounds a number to the specified number of significant digits, retaining trailing zeros.

        Args:
            number (float): The number to round.
            digits (int): The number of significant digits to retain.

        Returns:
            tuple: (scientific notation, decimal notation)
        """
        if number == 0:
            return f"{0:.{digits-1}e}", f"0.{'0' * (digits-1)}"
        elif digits <= 0:
            raise ValueError("The number of significant digits must be greater than 0.")

        # Determine the scale of the number
        scale = int(f"{number:.1e}".split('e')[1])  # Extract exponent using scientific notation

        # Calculate the factor to scale the number to 1 ≤ abs(value) < 10
        scale_factor = 10 ** (scale - digits + 1)

        # Round to nearest significant digit
        rounded = round(number / scale_factor) * scale_factor

        # Format the number with the required significant digits
        scientific_fmt = f"{rounded:.{digits-1}e}"  # Scientific notation format

        # Convert to plain decimal format
        decimal_str = f"{rounded:.{digits}f}"  # Force plain decimal format
        if '.' in decimal_str:
            integer_part, fractional_part = decimal_str.split('.')
            fractional_part = fractional_part.rstrip('0')  # Remove unnecessary trailing zeros
            decimal_fmt = f"{integer_part}.{fractional_part}" if fractional_part else integer_part
        else:
            decimal_fmt = decimal_str  # If no fractional part, keep as-is

        # Ensure decimal format always includes a period if it doesn't exist
        decimal_fmt = self.ensure_period(decimal_fmt)

        # decimal_fmt does not work:
        # What is 0.002150548826842929150265915084 in 5 significant digits?
        # Always include the decimal and answer with just the number.
        # 2.1505e-03
        # 0.00215

        return scientific_fmt, decimal_fmt

    #===========================================================================
    # Generic


# TEST

#problems = significantFigures()
#problems.print_problems()
