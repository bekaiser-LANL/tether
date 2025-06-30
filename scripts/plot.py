import os
import numpy as np
import matplotlib # pylint: disable=import-outside-toplevel
matplotlib.use('Agg') # pylint: disable=import-outside-toplevel
import matplotlib.pyplot as plt # pylint: disable=import-outside-toplevel

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")
graded_path = os.path.join(data_path, 'graded')
figure_path = os.path.join(data_path,'figures')

#==============================================================================
#==============================================================================

def load_data(graded_path, test_name, model_name_and_run):
    """
    Loads a .npz file and returns a dictionary of copied arrays.
    
    Args:
        graded_path (str): Directory path where the .npz file is stored.
        test_name (str): Name of the test or prefix for the file.
        model_name_and_run (str): Identifier for the model and run.

    Returns:
        dict: Dictionary containing copies of arrays from the .npz file.
    """
    model_path = os.path.join(graded_path,model_name_and_run.split('_')[0])
    filename = os.path.join(model_path, f"{test_name}_{model_name_and_run}_final_grade.npz")
    
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The file '{filename}' does not exist.")
    
    with np.load(filename, allow_pickle=True) as data:
        data_dict = {key: data[key].copy() for key in data.files}

    # indices for each solution category
    idx_A = [i for i, s in enumerate(data_dict['solution']) if s == 'A']
    idx_B = [i for i, s in enumerate(data_dict['solution']) if s == 'B']
    idx_C = [i for i, s in enumerate(data_dict['solution']) if s == 'C']

    # indices for each solution category
    idx_easy = [i for i, s in enumerate(data_dict['difficulty']) if s == 'easy']
    idx_medm = [i for i, s in enumerate(data_dict['difficulty']) if s == 'medium']
    idx_hard = [i for i, s in enumerate(data_dict['difficulty']) if s == 'hard']

    def slice_dict(d, idx):
        out = {}
        for k, v in d.items():
            sliced = v[idx]                      # NumPy array or list after slicing
            # Convert NumPy scalars to native Python types
            if isinstance(sliced, np.ndarray):
                sliced = sliced.astype(object).tolist()
            else:                                # already a list
                sliced = [x.item() if isinstance(x, np.generic) else x for x in sliced]
            out[k] = sliced
        return out

    data_dict_A = slice_dict(data_dict, idx_A)
    data_dict_B = slice_dict(data_dict, idx_B)
    data_dict_C = slice_dict(data_dict, idx_C)

    data_dict_easy = slice_dict(data_dict, idx_easy)
    data_dict_medm = slice_dict(data_dict, idx_medm)
    data_dict_hard = slice_dict(data_dict, idx_hard)

    return data_dict, data_dict_A, data_dict_B, data_dict_C, data_dict_easy, data_dict_medm, data_dict_hard

def grade_err(grade_estimate):
    n = len(grade_estimate)
    k = np.sum(grade_estimate)
    p = k/n 
    se = np.sqrt(p*(1-p)/n)
    upper = 1.96*se
    # if upper > 1.:
    #     upper = 1.
    lower = 1.96*se
    # if lower < 0.:
    #     lower = 0.
    return [lower,upper]

def grade_from_percentage(percent):
    if percent > 90.0:
        return 'A'
    elif 80.0 <= percent <= 89.9:
        return 'B'
    elif 70.0 <= percent <= 79.9:
        return 'C'
    elif 60.0 <= percent <= 69.9:
        return 'D'
    elif 33.3 <= percent <= 59.9:
        return 'F'
    elif percent < 33.3:
        return 'WTR'
    else:
        return 'Invalid'

def bar_plot(ax_, groups, errors, sub_labels, printed_sub_labels, exam_name):

    bar_width = 0.7
    fs = 17
    color1 = 'lightsteelblue'
    bar_width = 0.22
    
    n_groups = len(groups)
    n_sub = len(sub_labels)
    group_names = list(groups.keys())
    group_positions = np.arange(n_groups)
    all_results = []

    for i, group in enumerate(group_names):
        values = [groups[group][sub] for sub in sub_labels]
        start_x = group_positions[i] - (n_sub * bar_width) / 2 + bar_width / 2
        error_bounds = [errors[group][sub] for sub in sub_labels]  
        for j, (sub, val) in enumerate(zip(sub_labels, values)):
            x = start_x + j * bar_width
            lower_err, upper_err = error_bounds[j]
            if sub == "Total":
                # For the "All" sub-bar, use a different color scheme.
                lower_color = 'darkblue'
                upper_color = 'white'
            else:
                lower_color = color1
                upper_color = 'white'
            # Plot the lower portion from 0 to value.
            ax_.bar(x, val, width=bar_width, color=lower_color)
            # Plot the upper portion from value to 100.
            ax_.bar(x, 100 - np.array(val), width=bar_width, bottom=val, color=upper_color)
            # Label the sub-bar below the bar.
            #ax_.text(x, -0.03, sub, ha='center', va='top', fontsize=fs)
            ax_.text(x, -0.03, printed_sub_labels[j], ha='center', va='top', fontsize=fs)

            #ax1.errorbar(x, 100, yerr=err, fmt='none', ecolor='black', capsize=5, linewidth=1)
            # Error bars centered on the top of the bar
            ax_.errorbar(
                x, val,
                yerr=[[lower_err], [upper_err]],
                fmt='none', ecolor='black', capsize=5, linewidth=1
            )

            # Grading:
            if printed_sub_labels[0] != '$\\Delta_1$':
                if printed_sub_labels[j] not in ['A', 'B', 'C']:
                    # print('\n ',printed_sub_labels[j])
                    # print(' model = ',group)
                    # print(' value = ',val+upper_err)
                    # print(' grade = ',grade_from_percentage((val+upper_err)*100.))
                    # print(exam_name)
                    result = {
                        'sub_label': printed_sub_labels[j],
                        'model': group,
                        'value': val + upper_err,
                        'upper': upper_err,
                        'lower': lower_err,
                        'grade': grade_from_percentage((val + upper_err) * 100.0),
                        'exam_name': exam_name
                    }
                    all_results.append(result)
            

        # Add a group label below the sub-bar labels.
        ax_.text(group_positions[i], -0.11 ,group, ha='center', va='top', 
                fontsize=fs, fontweight='bold')

    return all_results

class GetData():

    def __init__(self, exam, exam_idx, model, model_idx, graded_path): #, **kwargs):
        self.model_name = model
        self.model_idx = model_idx
        self.benchmark_name = exam
        self.benchmark_idx = exam_idx
        self.graded_path = graded_path

        tmp = load_data(graded_path,
                        self.benchmark_name + '_' + self.benchmark_idx,
                        self.model_name + '_' + self.model_idx
                        )

        self.total = tmp[0]
        self.A = tmp[1]
        self.B = tmp[2]
        self.C = tmp[3]
        self.easy = tmp[4]
        self.medium = tmp[5]
        self.hard = tmp[6]

    def scores(self):
        return self.total['grade_estimate']

    def total_score(self):
        return np.sum(self.total['grade_estimate'])/len(self.total['grade_estimate'])
    
    def total_error(self):
        return grade_err(self.total['grade_estimate'])

    def A_scores(self):
        return self.A['grade_estimate']
    
    def A_score(self):
        return np.sum(self.A['grade_estimate'])/len(self.A['grade_estimate']) 

    def A_error(self):
        return grade_err(self.A['grade_estimate']) 

    def B_score(self):
        return np.sum(self.B['grade_estimate'])/len(self.B['grade_estimate'])

    def B_error(self):
        return grade_err(self.B['grade_estimate'])  

    def C_score(self):
        return np.sum(self.C['grade_estimate'])/len(self.C['grade_estimate'])

    def C_error(self):
        return grade_err(self.C['grade_estimate'])  

    def easy_score(self):
        return np.sum(self.easy['grade_estimate'])/len(self.easy['grade_estimate'])

    def easy_error(self):
        return grade_err(self.easy['grade_estimate'])
    
    def medium_score(self):
        return np.sum(self.medium['grade_estimate'])/len(self.medium['grade_estimate'])

    def medium_error(self):
        return grade_err(self.medium['grade_estimate'])  

    def hard_score(self):
        return np.sum(self.hard['grade_estimate'])/len(self.hard['grade_estimate'])

    def hard_error(self):
        return grade_err(self.hard['grade_estimate'])  

def extract_group_and_error_data(instances, name_mapping=None):
    """
    Extracts score and error data from a list of GetData class instances.

    Args:
        instances (list of tuples): Each tuple should be (label, class_instance)
        name_mapping (dict, optional): Optional mapping to rename model labels

    Returns:
        tuple: (groups1, groups2, errors1, errors2)
    """
    groups1, errors1 = {}, {}
    groups2, errors2 = {}, {}

    for label, model in instances:
        # Use mapped label if provided
        key = name_mapping[label] if name_mapping and label in name_mapping else label

        groups1[key] = {
            "A": model.A_score(),
            "B": model.B_score(),
            "C": model.C_score(),
            "Total": model.total_score()
        }

        errors1[key] = {
            "A": model.A_error(),
            "B": model.B_error(),
            "C": model.C_error(),
            "Total": model.total_error()
        }

        groups2[key] = {
            "easy": model.easy_score(),
            "medium": model.medium_score(),
            "hard": model.hard_score(),
            "Total": model.total_score()
        }

        errors2[key] = {
            "easy": model.easy_error(),
            "medium": model.medium_error(),
            "hard": model.hard_error(),
            "Total": model.total_error()
        }

    return groups1, groups2, errors1, errors2

def make_grade_plot(groups1, 
              groups2, 
              errors1, 
              errors2, 
              figure_path, 
              exam_name, 
              exam_title, 
              fs
              ):
    n_groups1 = len(groups1)
    n_groups2 = len(groups2)

    figname = figure_path + exam_name + '_ABC.png'
    fig = plt.figure(figsize=(16, 6))
    ax1=plt.subplot(1,1,1)
    grades = bar_plot(ax1, groups1, errors1, ["A", "B", "C", "Total"], ["A", "B", "C", "All"], exam_name)
    plt.axis([-0.5,n_groups1-0.5,0, 1])
    plt.ylabel('Percent correct',fontsize=fs+1)
    plt.title(exam_title,fontsize=fs)
    plt.yticks([0,0.33,1.],[r'$0$',r'$33$',r'$100$'],fontsize=fs)
    plt.plot(np.linspace(-0.5,n_groups1-0.5,num=100),np.ones([100])*0.33,color='black')
    ax1.tick_params(axis='x', which='both', bottom=False, top=False, left=False, labelbottom=False)
    plt.subplots_adjust(top=0.925, bottom=0.14, left=0.06, right=0.99, hspace=0.15, wspace=0.06)
    plt.savefig(figname,format="png"); plt.close(fig)

    figname = figure_path + exam_name + '_delta.png'
    fig = plt.figure(figsize=(16, 6)) 
    ax2=plt.subplot(1,1,1)
    tmp = bar_plot(ax2, groups2, errors2, ["easy", "medium", "hard", "Total"], [r"$\Delta_1$", r"$\Delta_2$", r"$\Delta_3$", "All"], exam_name)
    plt.axis([-0.5,n_groups2-0.5,0, 1])
    plt.ylabel('Percent correct',fontsize=fs+1)
    plt.title(exam_title,fontsize=fs)
    plt.yticks([0,0.33,1.],[r'$0$',r'$33$',r'$100$'],fontsize=fs)
    plt.plot(np.linspace(-0.5,n_groups2-0.5,num=100),np.ones([100])*0.33,color='black')
    ax2.tick_params(axis='x', which='both', bottom=False, top=False, left=False, labelbottom=False)
    plt.subplots_adjust(top=0.925, bottom=0.14, left=0.06, right=0.99, hspace=0.15, wspace=0.06)
    plt.savefig(figname,format="png"); plt.close(fig)

    # for result in grades:
    #     print(result)

    # save grades:
    cleaned_grades = []
    for result in grades:
        cleaned_result = result.copy()
        for key in ['value', 'upper', 'lower']:
            if key in cleaned_result:
                cleaned_result[key] = float(cleaned_result[key])
        cleaned_grades.append(cleaned_result)

    import pandas as pd
    df = pd.DataFrame(grades)
    print(df)
    df.to_csv('./' + exam_name + '_grades.csv', index=False)

def get_model_instances(exam_name,graded_path):
    """ Add new models here! """
    # Your initialized class instances
    granite    = GetData(exam_name, '0', 'granite3.2', '0', graded_path)
    mistral    = GetData(exam_name, '0', 'mistral', '0', graded_path)
    o3         = GetData(exam_name, '0', 'o3', '0', graded_path)
    claude3p7  = GetData(exam_name, '0', 'claude-3-7-sonnet-20250219', '0', graded_path)
    gpt4p1     = GetData(exam_name, '0', 'gpt-4.1', '0', graded_path)
    wizardmath = GetData(exam_name, '0', 'wizard-math', '0', graded_path)
    llama3p2   = GetData(exam_name, '0', 'llama3.2-1b', '0', graded_path)
    qwen2math  = GetData(exam_name, '0', 'qwen2-math', '0', graded_path)
    mathstral  = GetData(exam_name, '0', 'mathstral', '0', graded_path)
    o4mini     = GetData(exam_name, '0', 'o4-mini', '0', graded_path)

    # List of (label, instance) pairs
    model_instances = [
        ("granite3.2", granite),
        ("mistral", mistral),
        ("o3", o3),
        ("claude3.7", claude3p7),
        ("gpt-4.1", gpt4p1),
        ("wizard-math",wizardmath),
        ("llama3.2",llama3p2),
        ("qwen2-math",qwen2math),
        ("mathstral",mathstral),
        ("o4-mini",o4mini),        
    ]
    return model_instances
#==============================================================================
#==============================================================================

# SimpleInequality_tdist plots.

exam_name = 'SimpleInequality_tdist'
exam_title = 'Simple inequality, t-distribution, no solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)

#==============================================================================
#==============================================================================

# SimpleInequality_bootstrap plots.

exam_name = 'SimpleInequality_bootstrap'
exam_title = 'Simple inequality, bootstrap, no solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)


#==============================================================================
#==============================================================================

# SimpleInequalityWithMethod_tdist plots.

exam_name = 'SimpleInequalityWithMethod_tdist'
exam_title = 'Simple inequality, t-distribution, solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)

#==============================================================================
#==============================================================================

# SimpleInequalityWithMethod_bootstrap plots.

exam_name = 'SimpleInequalityWithMethod_bootstrap'
exam_title = 'Simple inequality, bootstrap, solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)

#==============================================================================
#==============================================================================

# MediatedCausalityWithMethod_tdist plots.

exam_name = 'MediatedCausalityWithMethod_tdist'
#exam_title = 'Mediated causality, t-distribution, solution method in prompt'
exam_title = 'Complex inequality, t-distribution, solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)

#==============================================================================
#==============================================================================

# MediatedCausalityWithMethod_bootstrap plots.

exam_name = 'MediatedCausalityWithMethod_bootstrap'
#exam_title = 'Mediated causality, bootstrap, solution method in prompt'
exam_title = 'Complex inequality, bootstrap, solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)

#==============================================================================
#==============================================================================

# MediatedCausality_tdist plots.

exam_name = 'MediatedCausality_tdist'
#exam_title = 'Mediated causality, t-distribution, no solution method in prompt'
exam_title = 'Complex inequality, t-distribution, no solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)


#==============================================================================
#==============================================================================

# MediatedCausality_bootstrap plots.

exam_name = 'MediatedCausality_bootstrap'
#exam_title = 'Mediated causality, bootstrap, no solution method in prompt'
exam_title = 'Complex inequality, bootstrap, no solution method in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)


#==============================================================================
#==============================================================================

# MediatedCausality_tdist plots.

exam_name = 'MediatedCausalitySmoking_tdist'
#exam_title = 'Mediated causality, t-distribution, no solution method in prompt, ontology'
exam_title = 'Complex inequality, t-distribution, no solution method in prompt, ontology in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)


#==============================================================================
#==============================================================================

# MediatedCausality_bootstrap plots.

exam_name = 'MediatedCausalitySmoking_bootstrap'
#exam_title = 'Mediated causality, bootstrap, no solution method in prompt, ontology in prompt'
exam_title = 'Complex inequality, bootstrap, no solution method in prompt, ontology in prompt'
fs = 17 # fontsize

model_instances = get_model_instances(exam_name,graded_path)
groups1, groups2, errors1, errors2 = extract_group_and_error_data(model_instances)
make_grade_plot(groups1, groups2, errors1, errors2, figure_path, exam_name, exam_title, fs)
