class SortByAnswer():

    def __init__(self, difficulty_thresholds, n_problems):

        self.difficulty_thresholds = difficulty_thresholds
        self.continue_flag = False
        self.diff_flag = 'none'
        self.n_problems = n_problems

    def initialize(self):
        # difference, difficulty flag, continue flag (True=continue on to next example)
        return [], self.diff_flag, self.continue_flag

    def update_difficulty(self,diff_value):
        if diff_value <= self.difficulty_thresholds[0]:
            self.diff_flag = 'hard'
        elif diff_value <= self.difficulty_thresholds[1]:
            self.diff_flag = 'medm'
        elif diff_value > self.difficulty_thresholds[1]:
            self.diff_flag = 'easy'
        return self.diff_flag

    def no_more_hard_problems_needed(self,hard):
        if self.diff_flag == 'hard' and hard["n_problems"] >= int(self.n_problems/3):
            # if enough problems have been sampled already,
            # then continue to another example
            return True
        return False

    def no_more_medm_problems_needed(self,medm):
        if self.diff_flag == 'medm' and medm["n_problems"] >= int(self.n_problems/3):
            # if enough medium problems have been sampled already,
            # then continue to another example
            return True
        return False

    def no_more_easy_problems_needed(self,easy):
        if self.diff_flag == 'easy' and easy["n_problems"] >= int(self.n_problems/3):
            # if enough easy problems have been sampled already,
            # then continue to another example
            return True
        return False

    def get_diff_flag(self):
        return self.diff_flag
    