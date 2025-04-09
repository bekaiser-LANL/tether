
def create_missing_directory(directory_path):
    import os
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def is_divisible_by_9(number):
    return number % 9 == 0

def is_divisible_by_3(number):
    return number % 3 == 0