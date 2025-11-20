import pickle
from pathlib import Path
import math
import csv
from ngram_model_def import NGramLanguageModel

TRAIN_PATH = "train.txt"
VAL_PATH = "val.txt"
TEST_PATH = "test.txt"
PICKLE_PATH = Path(__file__).with_name('ngram_models.pkl')
OUTPUT_CSV = Path(__file__).with_name('pmi_top100.csv')
MIN_COUNT = 5

def load_models(pickle_path):
    with open(pickle_path, 'rb') as f:
        models = pickle.load(f)
    return models

def get_counts(models):
    unigram = models.get('Unigram')
    bigram = models.get('Bigram')
    uni_counts = unigram.ngram_counts
    bi_counts = bigram.ngram_counts
    total_unigrams = sum(uni_counts.values())
    total_bigrams = sum(bi_counts.values())
    return uni_counts, bi_counts, total_unigrams, total_bigrams

def compute_pmi(uni_counts, bi_counts, total_unigrams, total_bigrams, min_count=MIN_COUNT):
    # count upon total formula use kairi che for probablities
    results = []
    for (w1, w2), c_xy in bi_counts.items():
        if c_xy < min_count:
            continue # skip low count bigrams to reduce noise
        c_x = uni_counts.get((w1,), 0)
        c_y = uni_counts.get((w2,), 0)
        if c_x == 0 or c_y == 0:
            continue
        p_x = c_x / total_unigrams
        p_y = c_y / total_unigrams
        p_xy = c_xy / total_bigrams
        denom = p_x * p_y
        if denom == 0 or p_xy == 0:
            continue
        pmi = math.log(p_xy / denom)
        results.append((w1, w2, c_xy, c_x, c_y, p_x, p_y, p_xy, pmi))

    results.sort(key=lambda x: x[-1], reverse=True)
    return results

def write_csv(results, out_path, top_k=100):
    headers = ['word1', 'word2', 'count_bigram', 'count_w1', 'count_w2', 'p_w1', 'p_w2', 'p_w1w2', 'pmi']
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in results[:top_k]:
            writer.writerow(row)

def read_sentences(filepath):
    with open(filepath, encoding='utf-8') as f:
        return [line.strip().split() for line in f if line.strip()]

def extract_bigrams_from_sentences(sentences):
    bigrams = []
    for tokens in sentences:
        bigrams.extend([(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)])
    return bigrams

def compute_pmi_for_bigrams(bigrams, uni_counts, bi_counts, total_unigrams, total_bigrams):
    pmi_scores = []
    for w1, w2 in bigrams:
        c_xy = bi_counts.get((w1, w2), 0)
        c_x = uni_counts.get((w1,), 0)
        c_y = uni_counts.get((w2,), 0)
        if c_x == 0 or c_y == 0 or c_xy == 0:
            pmi = None
        else:
            p_x = c_x / total_unigrams
            p_y = c_y / total_unigrams
            p_xy = c_xy / total_bigrams
            denom = p_x * p_y
            if denom == 0 or p_xy == 0:
                pmi = None
            else:
                pmi = math.log(p_xy / denom)
        pmi_scores.append((w1, w2, c_xy, c_x, c_y, pmi))
    return pmi_scores

def print_pmi_scores(pmi_scores, label, top_k=10):
    filtered = [row for row in pmi_scores if row[-1] is not None]
    filtered.sort(key=lambda x: x[-1], reverse=True)
    print(f"\nTop {top_k} PMI bigrams in {label}:")
    for i, row in enumerate(filtered[:top_k], 1):
        w1, w2, c_xy, c_x, c_y, pmi = row
        print(f"{i:2d}. {w1} {w2}  PMI={pmi:.4f}  count={c_xy}")

print('Loading models from', PICKLE_PATH)
models = load_models(PICKLE_PATH)
# Print summary for Unigram and Bigram models
for key in ['Unigram', 'Bigram']:
    model = models.get(key)
    if model is not None:
        print(f"{key} model: n={getattr(model, 'n', 'N/A')}, ngram_counts entries={len(getattr(model, 'ngram_counts', {}))}")
uni_counts, bi_counts, total_unigrams, total_bigrams = get_counts(models)
print('Total unigrams:', total_unigrams, 'Total bigrams:', total_bigrams)
results = compute_pmi(uni_counts, bi_counts, total_unigrams, total_bigrams)
if not results:
    print('No bigrams passed the minimum count threshold')
    exit(0)
write_csv(results, OUTPUT_CSV)
print(f'Wrote top {min(100, len(results))} PMI bigrams to', OUTPUT_CSV)
print('\nTop 10 PMI bigrams:')
for i, r in enumerate(results[:10], 1):
    w1, w2, c_xy, c_x, c_y, p_x, p_y, p_xy, pmi = r
    print(f"{i:2d}. {w1} {w2}  PMI={pmi:.4f}  count={c_xy}")

train_sentences = read_sentences(TRAIN_PATH)
unigram_model = NGramLanguageModel(n=1)
bigram_model = NGramLanguageModel(n=2)
unigram_model.train(train_sentences)
bigram_model.train(train_sentences)
uni_counts = unigram_model.ngram_counts
bi_counts = bigram_model.ngram_counts
total_unigrams = sum(uni_counts.values())
total_bigrams = sum(bi_counts.values())

for split, path in [('val', VAL_PATH), ('test', TEST_PATH)]:
    sentences = read_sentences(path)
    bigrams = extract_bigrams_from_sentences(sentences)
    pmi_scores = compute_pmi_for_bigrams(bigrams, uni_counts, bi_counts, total_unigrams, total_bigrams)
    print_pmi_scores(pmi_scores, split, top_k=10)