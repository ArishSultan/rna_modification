import RNA
import math
import nltk
import numpy as np
from pandas import DataFrame
from gensim.models import Word2Vec
from src.dataset import load_benchmark_dataset, Species, Modification


def word2vec_distance(seq1: str, seq2: str, k: int = 3, vector_size: int = 100) -> float:
    def seq_to_kmers(seq):
        return [seq[i:i + k] for i in range(len(seq) - k + 1)]

    kmers1, kmers2 = seq_to_kmers(seq1), seq_to_kmers(seq2)
    model = Word2Vec([kmers1 + kmers2], vector_size=vector_size, window=5, min_count=1, workers=4)

    vec1 = np.mean([model.wv[kmer] for kmer in kmers1], axis=0)
    vec2 = np.mean([model.wv[kmer] for kmer in kmers2], axis=0)

    return np.linalg.norm(vec1 - vec2)


def make_row(title: str, scores: dict) -> str:
    return (f'{title} & {scores['min']} & {scores['max']} & {scores['mean']} & {scores['median']} & {scores['std']} &'
            f' {scores['var']} & {scores['25%']} & {scores['50%']} & {scores['75%']} & {scores['same']} \\\\')


def make_section(title: str, scores: list) -> str:
    section = f"""
\\section*{{{title}}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lcccccccccc}}
\\toprule
Scenario & Min & Max & Mean & Median & Std & Variance & 25\\% & 50\\% & 75\\% & Same \\\\
\\midrule
{make_row("\\texttt{+ive to +ive}", scores[0])}
{make_row("\\texttt{+ive to -ive}", scores[1])}
{make_row("\\texttt{-ive to -ive}", scores[2])}
\\bottomrule
\\end{{tabular}}
\\caption{{{title} for Hamming Distance}}
\\end{{table}}
"""
    return section


def make_train_test_section(scores: list) -> str:
    section = f"""
\\section*{{Train-Test Analysis}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lcccccccccc}}
\\toprule
Scenario & Min & Max & Mean & Median & Std & Variance & 25\\% & 50\\% & 75\\% & Same \\\\
\\midrule
{make_row("\\texttt{T+ive to I+ive}", scores[0])}
{make_row("\\texttt{T-ive to I-ive}", scores[1])}
{make_row("\\texttt{T+ive to I-ive}", scores[2])}
{make_row("\\texttt{T-ive to I+ive}", scores[3])}
\\bottomrule
\\end{{tabular}}
\\caption{{Train-Test analysis for Hamming Distance}}
\\end{{table}}
"""
    return section


def make_distance(title: str, train_scores: list, test_scores: list, train_test_scores: list) -> str:
    train_section = make_section("Train Data Analysis", train_scores)
    test_section = make_section("Test Data Analysis", test_scores)
    train_test_section = make_train_test_section(train_test_scores)

    return f"""
\\newpage
\\begin{{center}}
\\textbf{{\\Huge{{{title}}}}}
\\end{{center}}
\\vspace{{30pt}}
{train_section}
{test_section}
{train_test_section}
"""


def generate_latex_document(distance_analyses: dict) -> str:
    header = r"""\documentclass{article}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage{array}
\usepackage{float}
\usepackage{booktabs}
\usepackage{courier}
\geometry{a4paper, margin=1in}
\begin{document}
"""
    footer = r"""
\end{document}
"""
    body = ""
    for title, (train_scores, test_scores, train_test_scores) in distance_analyses.items():
        body += make_distance(title, train_scores, test_scores, train_test_scores)

    return header + body + footer


def hamming_distance(seq1, seq2):
    return sum(ch1 != ch2 for ch1, ch2 in zip(seq1, seq2))


def levenshtein_distance(seq1: str, seq2: str) -> int:
    return nltk.edit_distance(seq1, seq2)


def needleman_wunsch_distance(seq1: str, seq2: str) -> int:
    gap_penalty = -1
    match_score = 1
    mismatch_penalty = -1

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i * gap_penalty
    for j in range(n + 1):
        dp[0][j] = j * gap_penalty

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(match, delete, insert)

    return dp[m][n]


def smith_waterman_distance(seq1: str, seq2: str) -> int:
    gap_penalty = -1
    match_score = 2
    mismatch_penalty = -1

    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    max_score = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match = dp[i - 1][j - 1] + (match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty)
            delete = dp[i - 1][j] + gap_penalty
            insert = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(0, match, delete, insert)
            max_score = max(max_score, dp[i][j])

    return max_score


def jukes_cantor_distance(seq1: str, seq2: str) -> float:
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")

    mismatches = sum(el1 != el2 for el1, el2 in zip(seq1, seq2))
    p = mismatches / len(seq1)

    if p == 0:
        return 0.0

    return -0.75 * math.log(1 - (4 / 3) * p)


import numpy as np
from scipy.spatial.distance import euclidean


def cgr_distance(seq1: str, seq2: str) -> float:
    def cgr(seq):
        x, y = 0.5, 0.5
        points = []
        for nucleotide in seq:
            if nucleotide == 'A':
                x, y = x / 2, y / 2
            elif nucleotide == 'C':
                x, y = x / 2, (1 + y) / 2
            elif nucleotide == 'G':
                x, y = (1 + x) / 2, (1 + y) / 2
            elif nucleotide == 'U':
                x, y = (1 + x) / 2, y / 2
            points.append((x, y))
        return np.array(points)

    cgr1, cgr2 = cgr(seq1), cgr(seq2)
    return euclidean(np.mean(cgr1, axis=0), np.mean(cgr2, axis=0))


def kimura_distance(seq1: str, seq2: str) -> float:
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of the same length")

    transitions = 0
    transversions = 0

    transitions_pairs = [('A', 'G'), ('G', 'A'), ('C', 'U'), ('U', 'C')]

    for el1, el2 in zip(seq1, seq2):
        if el1 != el2:
            if (el1, el2) in transitions_pairs:
                transitions += 1
            else:
                transversions += 1

    p = transitions / len(seq1)
    q = transversions / len(seq1)

    if p + q >= 0.75:
        raise ValueError("Invalid sequences, the total of p and q must be less than 0.75")

    return -0.5 * math.log(1 - 2 * p - q) - 0.25 * math.log(1 - 2 * q)


def gc_content_distance(seq1: str, seq2: str) -> float:
    def gc_content(seq: str) -> float:
        return (seq.count('G') + seq.count('C')) / len(seq)

    return abs(gc_content(seq1) - gc_content(seq2))


def base_pair_distance(seq1: str, seq2: str) -> int:
    def get_base_pairs(seq: str) -> set:
        structure, _ = RNA.fold(seq)
        return set(RNA.base_pair(seq, structure))

    bp1 = get_base_pairs(seq1)
    bp2 = get_base_pairs(seq2)

    return len(bp1.symmetric_difference(bp2))


def mountain_metric_distance(seq1: str, seq2: str) -> float:
    def mountain_vector(seq: str) -> list:
        structure, _ = RNA.fold(seq)
        return [sum([1 if i == '(' else -1 if i == ')' else 0 for i in structure[:k]]) for k in range(len(seq))]

    mv1 = mountain_vector(seq1)
    mv2 = mountain_vector(seq2)

    return sum((a - b) ** 2 for a, b in zip(mv1, mv2)) ** 0.5


def ensemble_distance(seq1: str, seq2: str) -> float:
    def ensemble_distance_vector(seq: str) -> list:
        fold_compound = RNA.fold_compound(seq)
        structure, mfe = fold_compound.mfe()
        pf = fold_compound.pf()
        return [fold_compound.bpp() for _ in range(len(seq))]

    ev1 = ensemble_distance_vector(seq1)
    ev2 = ensemble_distance_vector(seq2)

    return sum((a - b) ** 2 for a, b in zip(ev1, ev2)) ** 0.5

def rank_distance(u, v):
    # Create dictionaries to store the positions of characters in u and v
    pos_u = {}
    pos_v = {}

    for idx, char in enumerate(u):
        if char not in pos_u:
            pos_u[char] = []
        pos_u[char].append(idx + 1)

    for idx, char in enumerate(v):
        if char not in pos_v:
            pos_v[char] = []
        pos_v[char].append(idx + 1)

    # Calculate the rank distance
    rank_dist = 0

    # Calculate the sum of absolute differences for common characters
    common_chars = set(u) & set(v)
    for char in common_chars:
        for i in range(min(len(pos_u[char]), len(pos_v[char]))):
            rank_dist += abs(pos_u[char][i] - pos_v[char][i])

    # Add the positions of characters that are in u but not in v
    unique_to_u = set(u) - set(v)
    for char in unique_to_u:
        for pos in pos_u[char]:
            rank_dist += pos

    # Add the positions of characters that are in v but not in u
    unique_to_v = set(v) - set(u)
    for char in unique_to_v:
        for pos in pos_v[char]:
            rank_dist += pos

    return rank_dist


def calculate_distance(list1, list2, distance_func):
    distances = []
    for seq1 in list1:
        for seq2 in list2:
            distance = distance_func(seq1, seq2)
            distances.append(distance)
    return distances


def analyse_sets(seq_set1: list[str], seq_set2: list[str], distance_func):
    same = 0
    distances = []

    for item1 in seq_set1:
        for item2 in seq_set2:
            if item1 == item2:
                same += 1
                # Skip if both items are otherwise, since it does not make a sense.
                continue

            # Consider absolute distance in case of negative values.
            distances.append(distance_func(item1, item2))

    return {
        "same": same,
        "min": '{:.2f}'.format(np.min(distances)),
        "max": '{:.2f}'.format(np.max(distances)),
        "std": '{:.2f}'.format(np.std(distances)),
        "var": '{:.2f}'.format(np.var(distances)),
        "mean": '{:.2f}'.format(np.mean(distances)),
        "median": '{:.2f}'.format(np.median(distances)),
        "25%": '{:.2f}'.format(np.percentile(distances, 25)),
        "50%": '{:.2f}'.format(np.percentile(distances, 50)),
        "75%": '{:.2f}'.format(np.percentile(distances, 75)),
    }


def analyze_distances(train_data, test_data, distance_funcs):
    i_pos, i_neg = test_data
    t_pos, t_neg = train_data

    analysis_results = dict()
    for algorithm, algorithm_func in distance_funcs.items():
        print(algorithm)

        # Train analysis
        train_list = []
        print('  Analyzing Train +ive to Train +ive')
        train_list.append(analyse_sets(t_pos, t_pos, algorithm_func))
        print('  Analyzing Train +ive to Train -ive')
        train_list.append(analyse_sets(t_pos, t_neg, algorithm_func))
        print('  Analyzing Train -ive to Train -ive')
        train_list.append(analyse_sets(t_neg, t_neg, algorithm_func))

        # Test analysis
        test_list = []
        print('  Analyzing Test +ive to Test +ive')
        test_list.append(analyse_sets(i_pos, i_pos, algorithm_func))
        print('  Analyzing Test +ive to Test -ive')
        test_list.append(analyse_sets(i_pos, i_neg, algorithm_func))
        print('  Analyzing Test -ive to Test -ive')
        test_list.append(analyse_sets(i_neg, i_neg, algorithm_func))

        # Train-Test analysis
        train_test_list = []
        print('  Analyzing Train +ive to Test +ive')
        train_test_list.append(analyse_sets(t_pos, i_pos, algorithm_func))
        print('  Analyzing Train +ive to Test -ive')
        train_test_list.append(analyse_sets(t_neg, i_neg, algorithm_func))
        print('  Analyzing Train -ive to Test +ive')
        train_test_list.append(analyse_sets(t_pos, i_neg, algorithm_func))
        print('  Analyzing Train -ive to Test -ive')
        train_test_list.append(analyse_sets(t_neg, i_pos, algorithm_func))

        analysis_results[algorithm] = (train_list, test_list, train_test_list)

    return analysis_results

def remove_central_character(s):
    if len(s) % 2 == 0:
        raise ValueError("The input string must have an odd length.")

    central_index = len(s) // 2
    return s[:central_index] + s[central_index + 1:]

def extract_sequence(chunk: DataFrame):
    return list(map(remove_central_character, chunk['sequence'].values))


test_dataset = load_benchmark_dataset(Species.human, Modification.psi, True)
train_dataset = load_benchmark_dataset(Species.human, Modification.psi)

test_positives = extract_sequence(test_dataset.samples[test_dataset.targets == 1])
test_negatives = extract_sequence(test_dataset.samples[test_dataset.targets == 0])

train_positives = extract_sequence(train_dataset.samples[train_dataset.targets == 1])
train_negatives = extract_sequence(train_dataset.samples[train_dataset.targets == 0])

results = analyze_distances(
    (train_positives, train_negatives),
    (test_positives, test_negatives),
    {
        "Chaos Game Representation": cgr_distance,
        "Rank Distance": rank_distance,
        'Word2Vec': word2vec_distance,
        # "Hamming Distance": hamming_distance,
        # "Levenshtein Distance": levenshtein_distance,
        # "Needleman Wunsch": needleman_wunsch_distance,
        # "Smith Waterman": smith_waterman_distance,
        # "Jukes Cantor": jukes_cantor_distance,
        # "Kimura": kimura_distance,
        # "Tamura Nei": tamura_nei_distance,
        # "GC Content": gc_content_distance,
        # "Base Pair": base_pair_distance,
        # "Base Pair": base_pair_distance,
        # "Tree Edit": tree_edit_distance,
        # "Mountain Metric": mountain_metric_distance,
        # "Ensemble": ensemble_distance
    }
)

with open('test_12_Jul_results.tex', 'w') as f:
    f.write(generate_latex_document(results))
    f.close()

import subprocess

subprocess.run(['pdflatex', 'test_12_Jul_results.tex'])
