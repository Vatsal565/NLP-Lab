import math
import random
from collections import Counter, defaultdict
from pathlib import Path

START_TAG = "<START>"
END_TAG = "<END>"


def read_tagged_corpus(path):
    sentences = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = []
            tags = []
            for token in line.split():
                if "/" not in token:
                    continue
                word, tag = token.rsplit("/", 1)
                words.append(word)
                tags.append(tag)
            if words:
                sentences.append((words, tags))
    return sentences


def make_folds(data, k=5, seed=42):
    shuffled = list(data)
    random.Random(seed).shuffle(shuffled)
    fold_size = max(1, len(shuffled) // k)
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else len(shuffled)
        folds.append(shuffled[start:end])
    return folds


def train_hmm(train_data):
    tag_counts = Counter()
    emission_counts = defaultdict(Counter)
    transition_counts = defaultdict(Counter)
    vocabulary = set()

    for words, tags in train_data:
        prev_tag = START_TAG
        transition_counts[prev_tag]  # ensure key exists
        for word, tag in zip(words, tags):
            vocabulary.add(word)
            tag_counts[tag] += 1
            emission_counts[tag][word] += 1
            transition_counts[prev_tag][tag] += 1
            prev_tag = tag
        transition_counts[prev_tag][END_TAG] += 1

    tags = list(tag_counts.keys())
    vocab_size = len(vocabulary)
    num_tags = len(tags)

    log_transitions = defaultdict(dict)
    for prev_tag, next_counter in transition_counts.items():
        total = sum(next_counter.values())
        for candidate in tags + [END_TAG]:
            count = next_counter.get(candidate, 0)
            log_transitions[prev_tag][candidate] = math.log(
                (count + 1) / (total + num_tags + 1)
            )

    log_emissions = defaultdict(dict)
    log_unknown = {}
    for tag in tags:
        total = sum(emission_counts[tag].values())
        denom = total + vocab_size + 1
        log_unknown[tag] = math.log(1 / denom)
        for word, count in emission_counts[tag].items():
            log_emissions[tag][word] = math.log((count + 1) / denom)

    return {
        "tags": tags,
        "vocabulary": vocabulary,
        "log_transitions": log_transitions,
        "log_emissions": log_emissions,
        "log_unknown": log_unknown,
    }


def viterbi_decode(words, params):
    tags = params["tags"]
    log_trans = params["log_transitions"]
    log_emit = params["log_emissions"]
    log_unk = params["log_unknown"]

    dp = []
    backpointer = []

    first_dp = {}
    first_bp = {}
    for tag in tags:
        trans_score = log_trans[START_TAG].get(tag, math.log(1e-12))
        emit_score = log_emit[tag].get(words[0], log_unk[tag])
        first_dp[tag] = trans_score + emit_score
        first_bp[tag] = START_TAG
    dp.append(first_dp)
    backpointer.append(first_bp)

    for t in range(1, len(words)):
        curr_dp = {}
        curr_bp = {}
        word = words[t]
        for tag in tags:
            best_score = -math.inf
            best_prev = None
            emit_score = log_emit[tag].get(word, log_unk[tag])
            for prev_tag in tags:
                prev_score = dp[t - 1][prev_tag]
                trans_score = log_trans[prev_tag].get(tag, math.log(1e-12))
                score = prev_score + trans_score + emit_score
                if score > best_score:
                    best_score = score
                    best_prev = prev_tag
            curr_dp[tag] = best_score
            curr_bp[tag] = best_prev
        dp.append(curr_dp)
        backpointer.append(curr_bp)

    best_final_score = -math.inf
    best_last_tag = None
    for tag in tags:
        score = dp[-1][tag] + log_trans[tag].get(END_TAG, math.log(1e-12))
        if score > best_final_score:
            best_final_score = score
            best_last_tag = tag

    sequence = [best_last_tag]
    for t in range(len(words) - 1, 0, -1):
        prev_tag = backpointer[t][sequence[-1]]
        sequence.append(prev_tag)
    sequence.reverse()
    return sequence


def evaluate(predicted, gold):
    correct = sum(p == g for p, g in zip(predicted, gold))
    total = len(gold)
    precision = correct / total if total else 0.0
    recall = precision
    f1 = (2 * precision * recall / (precision + recall)) if precision else 0.0
    return precision, recall, f1


def cross_validate(sentences, k=5):
    folds = make_folds(sentences, k=k)
    fold_metrics = []

    for fold_idx in range(k):
        test_set = folds[fold_idx]
        train_set = [s for i, fold in enumerate(folds) if i != fold_idx for s in fold]
        params = train_hmm(train_set)

        all_pred = []
        all_gold = []
        for words, tags in test_set:
            if not words:
                continue
            pred_tags = viterbi_decode(words, params)
            all_pred.extend(pred_tags)
            all_gold.extend(tags)

        precision, recall, f1 = evaluate(all_pred, all_gold)
        fold_metrics.append((precision, recall, f1))
        print(
            f"Fold {fold_idx + 1}: Precision={precision:.4f} "
            f"Recall={recall:.4f} F1={f1:.4f}"
        )

    avg_precision = sum(p for p, _, _ in fold_metrics) / k
    avg_recall = sum(r for _, r, _ in fold_metrics) / k
    avg_f1 = sum(f for _, _, f in fold_metrics) / k
    print(
        f"\nAverage: Precision={avg_precision:.4f} "
        f"Recall={avg_recall:.4f} F1={avg_f1:.4f}"
    )

data_path = Path("wsj_pos_tagged_en.txt")
sentences = read_tagged_corpus(data_path)
cross_validate(sentences, k=5)