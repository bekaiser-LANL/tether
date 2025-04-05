import numpy as np

def get_95_CI(P,N): # <--- change name to reflect that this is for the t-distribution
    # MOVE TO GENERAL UQ FILE.
    # t distribution with standard error
    # (as opposed to using 2*SD)
    se = standard_error_for_proportion(P,N) 
    return P+1.96*se,P-1.96*se

def standard_error_for_proportion(P,N):
    # Brayer, Edward F. "Calculating the standard error of a proportion." 
    # Journal of the Royal Statistical Society Series C: Applied Statistics 6.1 (1957): 67-68.
    return np.sqrt((P*(1.-P))/N) 

def check_probability(P):
    if P > 1.:
        print('\n ERROR: Probability > 1')
    elif P < 0.:
        print('\n ERROR: Probability < 0')
    return

def enforce_probability_bounds(var):
    if var > 1.:
        var = 1.
    elif var < 0.:
        var = 0.
    return var