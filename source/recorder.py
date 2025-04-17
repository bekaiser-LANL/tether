import numpy as np
import os

class RecordBenchmark():

    def __init__(self, path, model, exam):
        self.path = path
        self.model = model
        self.exam = exam
        self.exam_name = self.exam["exam_name"]

    def create_missing_directory(self,directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    def write_header(self,file):
        #print(self.exam.metadata["Name"])
        file.write(str(self.exam.metadata["Name"]) + '\n')
        #if self.exam.metadata["Name"] == 'mediatedCausalitySmoking' or self.exam.metadata["Name"] == 'mediatedCausalitySmokingWithMethod':
        if self.exam.metadata["Name"].startswith("mediatedCausality"):
            file.write("=" * 40 + "\n")
            file.write("- %.1f percent X causes Y \n" %(100.*self.exam.metadata["A_count"]/self.exam.metadata["n_problems"]))
            file.write("- %.1f percent not X causes Y \n" %(100.*self.exam.metadata["B_count"]/self.exam.metadata["n_problems"]))
            file.write("- %.1f percent uncertain \n" %(100.*self.exam.metadata["C_count"]/self.exam.metadata["n_problems"]))
            file.write("- %i total questions \n" %(self.exam.metadata["n_problems"]))
            file.write("=" * 40 + "\n\n")
        elif self.exam.metadata["Name"] == 'significantFigures':
            file.write("=" * 40 + "\n")
            #file.write("- %i total questions \n" %(self.exam.metadata["n_problems"]))
            file.write("=" * 40 + "\n\n")
        elif self.exam.metadata["Name"] == 'standardDeviation':
            file.write("=" * 40 + "\n")
            #file.write("- %i total questions \n" %(self.exam.metadata["n_problems"]))
            file.write("=" * 40 + "\n\n")

    def write_problem_metadata(self,file,idx):
        if self.exam.metadata["Name"] == 'mediatedCausalitySmoking' or self.exam.metadata["Name"] == 'mediatedCausalitySmokingWithMethod':
            file.write('\n P(Y=1|do(X=1)) = %.3f +/- %.3f' % (self.exam.metadata["P_Y1doX1"][idx], self.exam.metadata["P_Y1doX1_CI"][idx]))
            file.write('\n P(Y=1|do(X=0)) = %.3f +/- %.3f' % (self.exam.metadata["P_Y1doX0"][idx], self.exam.metadata["P_Y1doX0_CI"][idx]))
        elif self.exam.metadata["Name"] == 'significantFigures':
            pass
        elif self.exam.metadata["Name"] == 'standardDeviation':
            pass

    def is_integer(self, value):
        return isinstance(value, int)

    def write_npz_report(self, report, grade, correct, response): 
        # saves completed exam as an npz

        if not np.isnan(report['checkpoints']):
            if self.is_integer(report['exam_idx']):
                self.filename = os.path.join(
                    str(self.path), 
                    str(self.model), 
                    f"{str(self.exam_name)}_{str(report['exam_idx'])}_chkpt{str(int(report['question_idx']+1))}.npz"
                )
            else:    
                self.filename = os.path.join(
                    str(self.path), 
                    str(self.model), 
                    f"{str(self.exam_name)}_chkpt{str(int(report['question_idx']+1))}.npz"
                )
        else:
            if self.is_integer(report['exam_idx']):
                self.filename = os.path.join(
                    str(self.path), 
                    str(self.model), 
                    f"{str(self.exam_name)}_{str(report['exam_idx'])}.npz"
                )
            else:    
                self.filename = os.path.join(
                    str(self.path), 
                    str(self.model), 
                    f"{str(self.exam_name)}.npz"
                )       
           
        self.create_missing_directory(self.path + self.model + '/')
  
        if self.exam_name == 'significantFigures' or self.exam_name == 'standardDeviation':
            np.savez(self.filename, 
                    exam_name = self.exam_name,
                    questions = report['questions'],
                    solutions = report['solutions'],   
                    model_str = report['model_str'],
                    exam_str = report['exam_str'],                 
                    temp_str = report['temp_str'],
                    effort_str = report['effort_str'],
                    counter = report['counter'],
                    n_problems = self.exam.metadata['n_problems'],
                    correct = correct,
                    grade = grade,
                    response = response
            )        
        elif self.exam_name == 'mediatedCausalitySmoking' or self.exam_name == 'mediatedCausalitySmokingWithMethod':
               np.savez(self.filename, 
                    exam_name = self.exam_name,
                    questions = report['questions'],
                    solutions = report['solutions'],   
                    model_str = report['model_str'],
                    exam_str = report['exam_str'],                 
                    temp_str = report['temp_str'],
                    effort_str = report['effort_str'],
                    counter = report['counter'],                    
                    n_problems = self.exam.metadata['n_problems'],     
                    correct = correct,
                    grade = grade,
                    response = response,                                 
                    dP = self.exam.metadata["dP"],
                    P_Y1doX1 = self.exam.metadata["P_Y1doX1"],
                    P_Y1doX0 = self.exam.metadata["P_Y1doX0"],
                    P_Y1doX1_CI = self.exam.metadata["P_Y1doX1_CI"],
                    P_Y1doX0_CI = self.exam.metadata["P_Y1doX0_CI"],
                    A_count = self.exam.metadata["A_count"],   
                    B_count = self.exam.metadata["B_count"],   
                    C_count = self.exam.metadata["C_count"]                                                                          
            )    

    def write_txt_report(self, report): 

        if self.is_integer(report['exam_idx']):
            #self.filename = self.path + self.model + '/' + self.exam_name + '_' + str(report['exam_idx']) + '.txt'
            self.filename = os.path.join(
                str(self.path), 
                str(self.model), 
                f"{str(self.exam_name)}_{str(report['exam_idx'])}.txt"
            )
            #print(self.filename)
        else:    
            #self.filename = self.path + self.model + '/' + self.exam_name + '.txt'
            self.filename = os.path.join(
                str(self.path), 
                str(self.model), 
                f"{str(self.exam_name)}.txt"
            )
           
        self.create_missing_directory(self.path + self.model + '/')

        self.grade        = report['grade']
        self.correct      = report['correct']
        self.response     = report['response']
        self.questions    = report['questions']
        self.solutions    = report['solutions']
        self.question_idx = report['question_idx']
        self.model_str    = report['model_str']
        self.exam_str     = report['exam_str']
        self.length_str   = report['length_str']
        self.temp_str     = report['temp_str']    
        self.effort_str     = report['effort_str']   

        if self.question_idx == 0:

            file = open(self.filename, "w")
            self.write_header(file)
            file.close()

        with open(self.filename, "a") as file:
            file.write('\n\n ' + str(self.question_idx) + ') ' + self.questions[self.question_idx])
            #file.write(self.model_str + ' response: ' + self.response + '| Solution: ' + str(self.solutions[self.question_idx]) + ' | correct? ' + str(self.correct))
            file.write('\n' + self.model_str) # + ' response: ' + self.response + '| Solution: ' + str(self.solutions[self.question_idx]) + ' | correct? ' + str(self.correct))
            file.write('\n Response: ' + self.response)
            file.write('\n Correct? ' + str(self.correct))
            file.write('\n Solution: ' + str(self.solutions[self.question_idx]))
            self.write_problem_metadata(file,self.question_idx)


        if self.question_idx == int(len(self.questions)-1): # write footer

            # GRADE EXAM
            non_nan_count = np.sum(~np.isnan(self.grade))
            self.grade = (np.nansum(self.grade) / non_nan_count)*100.
            unanswered_questions = len(self.questions) - non_nan_count

            file = open(self.filename, "a")
            file.write('\n\n ********************************************************* \n')
            file.write(self.model_str)
            file.write(self.exam_str)
            file.write(self.length_str)
            if self.model == self.model == "gpt-4.5-preview" or self.model == "gpt-4o":
                file.write(self.temp_str)
            elif self.model == "o3-mini" or self.model == "o1":
                file.write(self.effort_str)                    
            file.write('\n\n Number of unanswered questions = %i ' %unanswered_questions)
            file.write('\n Final grade (percent correct) = %.1f'  %self.grade)
            file.close()

    def save_blank_exam_npz(self, report): 
        """ saves blank exam as an npz """

        self.filename = report['reuse'] + '/' + self.exam_name + '_' + str(report['exam_idx']) + '.npz'
        self.create_missing_directory( self.path + '/')
  
        if self.exam_name.startswith('significantFigures') or self.exam_name.startswith('standardDeviation'):
            np.savez(self.filename, 
                    exam_name = self.exam_name,
                    questions = report['questions'],
                    solutions = report['solutions'],   
                    model_str = report['model_str'],
                    exam_str = report['exam_str'],                 
                    temp_str = report['temp_str'],
                    effort_str = report['effort_str'],
                    n_problems = self.exam.metadata['n_problems']
            )    
        elif self.exam_name.startswith("mediatedCausality"):        
               np.savez(self.filename, 
                    exam_name = self.exam_name,
                    questions = report['questions'],
                    solutions = report['solutions'],   
                    model_str = report['model_str'],
                    exam_str = report['exam_str'],                 
                    temp_str = report['temp_str'],
                    effort_str = report['effort_str'],
                    n_problems = self.exam.metadata['n_problems'],   
                    ci_method = self.exam.metadata['ci_method'],                   
                    p_diff = self.exam.metadata["p_diff"],
                    p_diff_ci_upper = self.exam.metadata["p_diff_ci_upper"],
                    p_diff_ci_lower = self.exam.metadata["p_diff_ci_lower"],
                    difficulty = self.exam.metadata["difficulty"],
                    A_count = self.exam.metadata["a_count"],   
                    B_count = self.exam.metadata["b_count"],   
                    C_count = self.exam.metadata["c_count"]                                                                          
            )       

    def save_blank_exam_txt(self, report): 
        # saves the blank exam as a .txt
        # (can create weird problems when reading large .txt)

        self.filename = report['reuse'] + '/' + self.exam_name + '_' + str(report['exam_idx']) + '.txt'
        self.create_missing_directory( self.path + '/')

        self.grade        = report['grade']
        self.correct      = report['correct']
        self.response     = report['response']
        self.questions    = report['questions']
        self.solutions    = report['solutions']
        self.question_idx = report['question_idx']
        self.model_str    = report['model_str']
        self.exam_str     = report['exam_str']
        self.length_str   = report['length_str']
        self.temp_str     = report['temp_str']    

        if self.question_idx == 0:

            file = open(self.filename, "w")
            self.write_header(file)
            file.close()

        with open(self.filename, "a") as file:
            file.write('\n\n ' + str(self.question_idx) + ') Problem: ' + self.questions[self.question_idx])
            #file.write('\n\n Response:  \n Solution: ' + str(self.solutions[self.question_idx]))
            file.write('\n\n Model: ')
            file.write('\n Response: ')
            file.write('\n Correct? ')
            file.write('\n Solution: ' + str(self.solutions[self.question_idx]))
            self.write_problem_metadata(file,self.question_idx)

        if self.question_idx == int(len(self.questions)-1): # write footer

            file = open(self.filename, "a")
            file.write('\n\n ********************************************************* \n')
            file.write(self.exam_str)
            file.write(self.length_str)
            file.close()


#===============================================================================

# Tests:
#problems = significantFigures()
#problems.print_problems()
