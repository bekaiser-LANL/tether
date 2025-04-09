import os




def create_missing_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def is_divisible_by_9(number):
    return number % 9 == 0