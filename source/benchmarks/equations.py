import numpy as np
import math as ma
import random
import os
from PIL import Image
import torch
from transformers import AutoProcessor


class equations():


    def __init__(self): #range=[-100,100], n_numbers = 20, decimal_places=4, n_problems=100):
        self.imgpath = '/lustre/scratch5/dmperez/Tether/source/benchmarks/equations/' 
        # Path to the local directory containing images
        self.n_problems = num_images = len([f for f in os.listdir(self.imgpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) 
        #self.decimal_places = decimal_places
        #self.range = range
        #self.n_numbers = n_numbers
        self.metadata = {
            "Name": 'Equations'
        }
        self.make_problems() # all tests need this


    def make_problems(self): # all tests need this
        self.questions = [] # all tests need this
        self.solutions = [] # all tests need this
        for i in range(0,self.n_problems): # all tests need this

            image, q_str = self.generate_question()
            self.questions = np.append(self.questions,q_str)

            ans_str = ''
            self.solutions = np.append(self.solutions,ans_str)


    def generate_question(self): # all tests need this
        # Get all image files in the directory (supports PNG, JPG, JPEG)
        image_files = [f for f in os.listdir(self.imgpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Iterate through images
        for img_file in image_files:
            img_path = os.path.join(self.imgpath, img_file)

            # Read image using PIL
            image = Image.open(img_path)

            if image is None:
                raise ValueError("Falied to load PNG")
                #print(f"Skipping: {img_file} (could not load)")
                #continue

            # Construct the question with image and add prompt
            q_str = f"Describe this equation with only 2 sentences. Do not repeat the question. Do not write more than 2 sentences: {image}"
            
            return image, q_str
        return image_files

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
