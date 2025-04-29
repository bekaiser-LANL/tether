""" Analyze benchmark results """
import os
import argparse
from source.analyzer import Analyzer
from source.utils import detect_duplicate_tables, load_saved_benchmark

# Prior to running pytest, you need to set your path with:
# export PATH_TO_BENCHMARKS=ENTER_YOUR_PATH_HERE
# where ENTER_YOUR_PATH_HERE needs to be replaced with your path.
data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze the completed benchmark for a specified model."
    )

    parser.add_argument(
        "benchmark_name",
        help="Name of the benchmark to run, including its index (e.g., MediatedCausality_tdist_0)"
    )

    parser.add_argument(
        "model_name",
        help="Name of the model to test (e.g., gpt-4o)"
    )

    parser.add_argument(
        "--path",
        default=data_path,
        help=f"Path to the benchmarks directory (default: from PATH_TO_BENCHMARKS or '{data_path}')"
    )

    args = parser.parse_args()

    # Analyzer(args.path, args.model_name, args.benchmark_name)
    # print(f"\n Analyses of benchmark '{args.benchmark_name}' for model '{args.model_name}' completed at: {args.path}")

    # Verify that each mediated causality benchmark has no duplicate problems:
    exam_names = ['MediatedCausality_bootstrap',
                  'MediatedCausality_tdist',
                  'MediatedCausalitySmoking_tdist', # <- has a 
                  'MediatedCausalityWithMethod_tdist'
                  ]
    for i in range(0,len(exam_names)):
        data = load_saved_benchmark(data_path + '/blank/',exam_names[i],0)
        has_duplicates, duplicate_pairs, n_problems = detect_duplicate_tables(data['table'])
        print(f"\n Benchmark: {exam_names[i]}"
            f"\n Duplicate tables detected: {has_duplicates}"
            f"\n Number of problems: {n_problems}")
        if has_duplicates:
            print(f" {duplicate_pairs} duplicate pairs found")


if __name__ == "__main__":
    main()



# 

# # path to the completed benchmarks you want to analyze:
# path        = '/Users/l281800/Desktop/benchmarks/completed/'
# figure_path = '/Users/l281800/Desktop/benchmarks/figures/'
# exam_idx    = 1

# benchmarks  = ['mediatedCausalitySmoking','mediatedCausalitySmokingWithMethod']

#===============================================================================

# benchmark = 'MediatedCausalitySmoking_tdist'

# path_to_benchmarks = '/Users/l281800/Desktop/benchmarks/'
# npz_filename = path_to_benchmarks + 'results/' + benchmark + '.npz'



# # Load the .npz file
# data = np.load(npz_filename, allow_pickle=True)

# # List all keys stored in the file
# print("\n\n\n Keys:", data.files)

# # Access individual arrays
# print("\n\n First question:\n ", data["question"][0])
# print("\n First response:\n ", data["responses"][0])
# print("\n First solution:\n ", data["solution"][0])
# print("\n\n Last question:\n ", data["question"][-1])
# print("\n Last response:\n ", data["responses"][-1])
# print("\n Last solution:\n ", data["solution"][-1])
# print("\n Last p_diff:\n ", data["p_diff"][-1])

#===============================================================================

# print('\n\n GPT-4o:')

# mCS_4o     = Analyzer(path,'gpt-4o',benchmarks[0],exam_idx).get_data()
# print('\n mCS A_score = ',mCS_4o['A_score'])
# print(' mCS B_score = ',mCS_4o['B_score'])
# print(' mCS C_score = ',mCS_4o['C_score'])
# print(' mCS total score = ',mCS_4o['total_score'])
# # print(mCS_4o['response'])
# # print(mCS_4o['solutions'])
# # print(mCS_4o['P_Y1doX1'])
# # print(mCS_4o['P_Y1doX1_CI'])
# # print(mCS_4o['questions'])


# # mCSWM_4o   = analyses(path,'4o',benchmarks[1]).get_data()
# # print('\n mCSWM A_score = ',mCSWM_4o['A_score'])
# # print(' mCSWM B_score = ',mCSWM_4o['B_score'])
# # print(' mCSWM C_score = ',mCSWM_4o['C_score'])
# # print(' mCSWM total score = ',mCSWM_4o['total_score'])


# # print('\n\n o3-mini:')

# # mCS_o3mini     = analyses(path,'o3-mini',benchmarks[0]).get_data()
# # print('\n mCS A_score = ',mCS_o3mini['A_score'])
# # print(' mCS B_score = ',mCS_o3mini['B_score'])
# # print(' mCS C_score = ',mCS_o3mini['C_score'])
# # print(' mCS total score = ',mCS_o3mini['total_score'])

# # mCSWM_o3mini     = analyses(path,'o3-mini',benchmarks[1]).get_data()

# # print('\n mCSWM A_score = ',mCSWM_o3mini['A_score'])
# # print(' mCSWM B_score = ',mCSWM_o3mini['B_score'])
# # print(' mCSWM C_score = ',mCSWM_o3mini['C_score'])
# # print(' mCSWM total score = ',mCSWM_o3mini['total_score'])



# print('\n\n Llama 3:')

# mCS_l3     = analyses(path,'llama3',benchmarks[0],exam_idx).get_data()
# print('\n mCS A_score = ',mCS_l3['A_score'])
# print(' mCS B_score = ',mCS_l3['B_score'])
# print(' mCS C_score = ',mCS_l3['C_score'])
# print(' mCS total score = ',mCS_l3['total_score'])

# # mCSWM_l3   = analyses(path,'llama3',benchmarks[1]).get_data()
# # print('\n mCSWM A_score = ',mCSWM_l3['A_score'])
# # print(' mCSWM B_score = ',mCSWM_l3['B_score'])
# # print(' mCSWM C_score = ',mCSWM_l3['C_score'])
# # print(' mCSWM total score = ',mCSWM_l3['total_score'])


# print('\n\n Llama 3.2:')
# # need to look at the llama 3.2 exams

# mCS_l3p2     = analyses(path,'llama3.2',benchmarks[0],exam_idx).get_data()
# print('\n mCS A_score = ',mCS_l3p2['A_score'])
# print(' mCS B_score = ',mCS_l3p2['B_score'])
# print(' mCS C_score = ',mCS_l3p2['C_score'])
# print(' mCS total score = ',mCS_l3p2['total_score'])

# mCSWM_l3p2   = analyses(path,'llama3.2',benchmarks[1]).get_data()
# print('\n mCSWM A_score = ',mCSWM_l3p2['A_score'])
# print(' mCSWM B_score = ',mCSWM_l3p2['B_score'])
# print(' mCSWM C_score = ',mCSWM_l3p2['C_score'])
# print(' mCSWM total score = ',mCSWM_l3p2['total_score'])


# Make plot with dP and n_samples.

# do Claude

# do DeepSeek


#=======

"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# Example data: each value is between 0 and 100.
values = [mCS_4o['total_score'], mCS_o3mini['total_score'], mCS_l3['total_score'], mCS_l3p2['total_score']]  # darkblue portion
valuesWM = [mCSWM_4o['total_score'], mCSWM_o3mini['total_score'], mCSWM_l3['total_score'], mCSWM_l3p2['total_score']]
#labels = [f'Bar {i+1}' for i in range(len(values))]
labels = ['4o','o3-mini','llama3','llama3.2']
x = np.arange(len(values))
bar_width = 0.7
#print(x,x[0]-0.5,x[-1]+0.5)

figname = fig_path + 'check.png'
fig = plt.figure(figsize=(12, 5))

ax1=plt.subplot(1,2,1)
plt.bar(x, values, color='darkblue', width=bar_width, label='Correct')
plt.bar(x, [100 - v for v in values], bottom=values, color='thistle', width=bar_width, label='Incorrect')
plt.axis([x[0]-0.5,x[-1]+0.5,0, 100])  # Extra room above 100 for aesthetics
plt.plot(np.linspace(x[0]-0.5,x[-1]+0.5,num=100),np.ones([100])*33,color='black')
plt.xticks(x, labels,fontsize=14)
plt.ylabel('Problems',fontsize=14)
plt.title('No solution method in prompt',fontsize=14)
plt.legend(loc=1,framealpha=1.,fontsize=12)
plt.yticks([0,33,100.],[r'$0$',r'$33$',r'$100$'],fontsize=14)

ax1=plt.subplot(1,2,2)
plt.bar(x, valuesWM, color='darkblue', width=bar_width, label='Correct')
plt.bar(x, [100 - v for v in valuesWM], bottom=valuesWM, color='thistle', width=bar_width, label='Incorrect')
plt.axis([x[0]-0.5,x[-1]+0.5,0, 100])  # Extra room above 100 for aesthetics
plt.plot(np.linspace(x[0]-0.5,x[-1]+0.5,num=100),np.ones([100])*33,color='black')
plt.xticks(x, labels,fontsize=14)
plt.ylabel('Problems',fontsize=14)
plt.title('Solution method in prompt',fontsize=14)
plt.legend(loc=1,framealpha=1.,fontsize=12)
plt.yticks([0,33,100.],[r'$0$',r'$33$',r'$100$'],fontsize=14)

plt.subplots_adjust(top=0.925, bottom=0.14, left=0.07, right=0.985, hspace=0.4, wspace=0.35)
plt.savefig(figname,format="png"); plt.close(fig);


# NOW DO THEM BY A,B,C Answers



fs = 15
color1 = 'lightsteelblue'

figname = fig_path + 'check2.png'

# Set up the figure.
fig, ax = plt.subplots(figsize=(16, 6))

# Define the width of each sub-bar.
bar_width = 0.22

# Labels for the sub-bars.
sub_labels = ["A", "B", "C", "Total"]

ax1=plt.subplot(1,2,1)


groups = {
    "4o": {"A": mCS_4o['A_score'], "B": mCS_4o['B_score'], "C": mCS_4o['C_score'], "Total": mCS_4o['total_score']},
    "o3-mini": {"A": mCS_o3mini['A_score'], "B": mCS_o3mini['B_score'], "C": mCS_o3mini['C_score'], "Total": mCS_o3mini['total_score']},
    "llama3": {"A": mCS_l3['A_score'], "B": mCS_l3['B_score'], "C": mCS_l3['C_score'], "Total": mCS_l3['total_score']},
    "llama3.2": {"A": mCS_l3p2['A_score'], "B": mCS_l3p2['B_score'], "C": mCS_l3p2['C_score'], "Total": mCS_l3p2['total_score']},
}

# Number of groups and sub-bars per group.
n_groups = len(groups)
n_sub = len(sub_labels)
group_names = list(groups.keys())
# Create an array for group positions along the x-axis.
group_positions = np.arange(n_groups)

# For each group, plot each sub-bar.
for i, group in enumerate(group_names):
    # Get the list of values in the order of sub_labels.
    values = [groups[group][sub] for sub in sub_labels]
    # Compute the starting x-position so that the sub-bars are centered around the group position.
    # The center of the group is group_positions[i].
    # We shift left by half the total width of the sub-bars.
    start_x = group_positions[i] - (n_sub * bar_width) / 2 + bar_width / 2
    # for j, val in enumerate(values):
    #     x = start_x + j * bar_width
    #     # Plot the green portion (from 0 to val).
    #     ax.bar(x, val, width=bar_width, color='darkblue')
    #     # Plot the red portion (from val to 100).
    #     ax.bar(x, 100 - val, width=bar_width, bottom=val, color='white')
    #     # Label the sub-bar (A, B, C, All) below the x-axis.
    #     ax.text(x, -5, sub_labels[j], ha='center', va='top', fontsize=14)
    for j, (sub, val) in enumerate(zip(sub_labels, values)):
        x = start_x + j * bar_width

        if sub == "Total":
            # For the "All" sub-bar, use a different color scheme.
            lower_color = 'darkblue'
            upper_color = 'white'
        else:
            lower_color = color1
            upper_color = 'white'

        # Plot the lower portion from 0 to value.
        ax1.bar(x, val, width=bar_width, color=lower_color)
        # Plot the upper portion from value to 100.
        ax1.bar(x, 100 - val, width=bar_width, bottom=val, color=upper_color)
        # Label the sub-bar below the bar.
        ax1.text(x, -5, sub, ha='center', va='top', fontsize=fs)

    # Add a group label below the sub-bar labels.
    ax1.text(group_positions[i], -12, group, ha='center', va='top',
            fontsize=fs, fontweight='bold')

# Adjust the limits and labels.
plt.axis([-0.5,n_groups-0.5,0, 100])
plt.ylabel('Percent correct',fontsize=fs+1)
plt.title('No solution method in prompt',fontsize=fs)
plt.yticks([0,33,100.],[r'$0$',r'$33$',r'$100$'],fontsize=fs)
plt.plot(np.linspace(-0.5,n_groups-0.5,num=100),np.ones([100])*33,color='black')
#ax.set_title("Grouped Bar Chart with Sub-Bars (Green: Value, Red: Remainder to 100)")
ax1.tick_params(axis='x', which='both', bottom=False, top=False, left=False, labelbottom=False)

#=========


ax2=plt.subplot(1,2,2)


groups = {
    "4o": {"A": mCSWM_4o['A_score'], "B": mCSWM_4o['B_score'], "C": mCSWM_4o['C_score'], "Total": mCSWM_4o['total_score']},
    "o3-mini": {"A": mCSWM_o3mini['A_score'], "B": mCSWM_o3mini['B_score'], "C": mCSWM_o3mini['C_score'], "Total": mCSWM_o3mini['total_score']},
    "llama3": {"A": mCSWM_l3['A_score'], "B": mCSWM_l3['B_score'], "C": mCSWM_l3['C_score'], "Total": mCSWM_l3['total_score']},
    "llama3.2": {"A": mCSWM_l3p2['A_score'], "B": mCSWM_l3p2['B_score'], "C": mCSWM_l3p2['C_score'], "Total": mCSWM_l3p2['total_score']},
}

# Number of groups and sub-bars per group.
n_groups = len(groups)
n_sub = len(sub_labels)
group_names = list(groups.keys())
# Create an array for group positions along the x-axis.
group_positions = np.arange(n_groups)

# For each group, plot each sub-bar.
for i, group in enumerate(group_names):
    # Get the list of values in the order of sub_labels.
    values = [groups[group][sub] for sub in sub_labels]
    # Compute the starting x-position so that the sub-bars are centered around the group position.
    # The center of the group is group_positions[i].
    # We shift left by half the total width of the sub-bars.
    start_x = group_positions[i] - (n_sub * bar_width) / 2 + bar_width / 2
    # for j, val in enumerate(values):
    #     x = start_x + j * bar_width
    #     # Plot the green portion (from 0 to val).
    #     ax.bar(x, val, width=bar_width, color='darkblue')
    #     # Plot the red portion (from val to 100).
    #     ax.bar(x, 100 - val, width=bar_width, bottom=val, color='white')
    #     # Label the sub-bar (A, B, C, All) below the x-axis.
    #     ax.text(x, -5, sub_labels[j], ha='center', va='top', fontsize=14)
    for j, (sub, val) in enumerate(zip(sub_labels, values)):
        x = start_x + j * bar_width

        if sub == "Total":
            # For the "All" sub-bar, use a different color scheme.
            lower_color = 'darkblue'
            upper_color = 'white'
        else:
            lower_color = color1
            upper_color = 'white'

        # Plot the lower portion from 0 to value.
        ax2.bar(x, val, width=bar_width, color=lower_color)
        # Plot the upper portion from value to 100.
        ax2.bar(x, 100 - val, width=bar_width, bottom=val, color=upper_color)
        # Label the sub-bar below the bar.
        ax2.text(x, -5, sub, ha='center', va='top', fontsize=fs)

    # Add a group label below the sub-bar labels.
    ax2.text(group_positions[i], -12, group, ha='center', va='top',
            fontsize=fs, fontweight='bold')

# Adjust the limits and labels.
plt.axis([-0.5,n_groups-0.5,0, 100])
#plt.ylabel('Percent correct',fontsize=15)
plt.title('Solution method in prompt',fontsize=fs)
plt.yticks([],[],fontsize=fs)
plt.plot(np.linspace(-0.5,n_groups-0.5,num=100),np.ones([100])*33,color='black')
#ax.set_title("Grouped Bar Chart with Sub-Bars (Green: Value, Red: Remainder to 100)")
ax2.tick_params(axis='x', which='both', bottom=False, top=False, left=False, labelbottom=False)


plt.subplots_adjust(top=0.925, bottom=0.14, left=0.06, right=0.985, hspace=0.15, wspace=0.08)
plt.savefig(figname,format="png"); plt.close(fig);


# import matplotlib.pyplot as plt
# low_N = np.power(10.,self.min_power10_sample_size)
# high_N = np.power(10.,self.max_power10_sample_size)
# figname = self.plot_path + 'case_%i.png' %j
# fig = plt.figure(figsize=(12, 5))
# ax1=plt.subplot(1,2,1)
# plt.fill_between(N_samples, P_Y1doX1l, P_Y1doX1u, color='royalblue', alpha=0.2, label="95% CI P(Y=1|do(X=1))")
# plt.fill_between(N_samples, P_Y1doX0l, P_Y1doX0u, color='crimson', alpha=0.2, label="95% CI P(Y=1|do(X=0))")
# plt.plot(N_samples,P_Y1doX1,color='royalblue',linewidth=1)
# plt.plot(N_samples,P_Y1doX0,color='crimson',linewidth=1)
# plt.legend(loc=1,fontsize=13,framealpha=1.)
# plt.xlabel(r'$N_{samples}$',fontsize=18)
# plt.ylabel(r'Probability',fontsize=16)
# ax1.set_xscale("log")
# plt.axis([low_N,high_N,0.,1.])
# ax1=plt.subplot(1,2,2)
# plt.plot(N_samples,causality,color='black',linestyle='None',marker='o',markersize=10,linewidth=2)
# plt.xlabel(r'$N_{samples}$',fontsize=18)
# ax1.set_xscale("log")
# plt.grid()
# plt.axis([low_N,high_N,-0.5,2.5])
# plt.yticks([0.,1.,2.],[r'Uncertain',r'$X$ causes $Y$',r'$\neg X$ causes $Y$'],fontsize=14)


"""
print('\n Analyses ran to completion! \n ')
