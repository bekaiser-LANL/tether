import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from wordcloud import WordCloud
from matplotlib.ticker import MaxNLocator
from itertools import chain

data_path = os.environ.get("PATH_TO_BENCHMARKS", "/default/path")

# List your .npz files here (or automate with glob if needed)
npz_files = [
#    data_path+'/completed/SimpleInequality_tdist_wizard-math_0.npz',
#    data_path+'/completed/SimpleInequality_bootstrap_WizardMath-7B-V1.1_0.npz',
#    data_path+'/completed/SimpleInequality_tdist_o3_0.npz',
#    data_path+'/completed/SimpleInequality_bootstrap_o3_0.npz',
#    data_path+'/completed/MediatedCausality_tdist_wizard-math_0.npz',
    data_path+'/completed/MediatedCausality_bootstrap_wizard-math_0.npz',
#    data_path+'/completed/MediatedCausality_tdist_o3_0.npz',
#    data_path+'/completed/MediatedCausality_bootstrap_o3_0.npz'
]

# Set the key name in the .npz file that holds the text data
text_key = 'responses'  # Adjust if your key is different

# Output directory for word clouds
output_dir_clouds = "wordclouds"
output_dir_histos = "term_histograms_by_file"
os.makedirs(output_dir_clouds, exist_ok=True)
os.makedirs(output_dir_histos, exist_ok=True)

stat_terms = {
    'mean', 'median', 'mode', 'variance', 'standard deviation', 'std',
    'normal', 'distribution', 'bootstrap', 'bayesian', 'prior', 'posterior',
    'inference', 'regression', 'anova', 't-test', 'p-value', 'confidence level',
    'interval', 'samples', 'bias', 'hypothesis', 'null', 'alternative',
    'resampling', 'permutation', 'estimator', 'likelihood', 'chi-square',
    'correlation', 'causality', 'dependence', 'independence', 'model',
    'fit', 'significance', 'z-score', 'effect size', 'predictor', 'tdist',
    'mle', 'maximum likelihood', 'confidence interval', 'distributional'
}
stat_terms = set(term.lower() for term in stat_terms)

def normalize_word(word):
    return re.sub(r'[^\w\-]', '', word).lower()

def normalize_text(text):
    return [
        normalize_word(w) for w in text.split()
        if len(normalize_word(w)) > 1 and not normalize_word(w).isdigit()
    ]

def black_color_func(*args, **kwargs):
    return "black"

# Process files again just for frequency counting
for filename in npz_files:
    try:
        data = np.load(filename, allow_pickle=True)
        if text_key not in data:
            print(f"Key '{text_key}' not found in {filename}. Skipping.")
            continue

        text_data = data[text_key]
        base_name = os.path.splitext(os.path.basename(filename))[0]

        if isinstance(text_data, np.ndarray) and text_data.ndim > 1:
            text_data = list(chain.from_iterable(text_data))

        total_responses = 0

        word_counter = Counter()        
        for response in text_data:
            if not response or not isinstance(response, (str, np.str_)):
                continue

            total_responses += 1
            words = normalize_text(str(response))
            seen_stat_words = {w for w in words if w in stat_terms}
            word_counter.update(seen_stat_words)

        print(f"{base_name}: Processed {total_responses} responses, {len(word_counter)} unique stat words")
        
        if not word_counter:
            print(f"No valid words found in {base_name}. Skipping.")
            continue

        # Keep only stat terms that actually occurred
        matched_stat_terms = {word: count for word, count in word_counter.items() if word in stat_terms}

        # --- Word Cloud ---
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            color_func=black_color_func
        ).generate_from_frequencies(matched_stat_terms)

        wc_output_path = os.path.join(output_dir_clouds, f"{base_name}_wordcloud.jpg")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.savefig(wc_output_path, format='jpeg', dpi=1200, bbox_inches='tight')
        plt.close()
        print(f"Saved word cloud → {wc_output_path}")

        sorted_items = word_counter.most_common()
        labels, values = zip(*sorted_items)

        plt.figure(figsize=(12, 6))
        plt.bar(labels, values)
        plt.xticks(rotation=45, ha='right', fontsize=20)
        plt.yticks(fontsize=20)
        plt.ylabel("Unique Response Count", fontsize=20)
        plt.title(f"Top Words – {base_name}", fontsize=20)
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout()

        hist_output_path = os.path.join(output_dir_histos, f"{base_name}_term_histogram.jpg")
        plt.savefig(hist_output_path, dpi=300)
        plt.close()
        print(f"Saved histogram → {hist_output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")
