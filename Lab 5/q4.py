"""
P(w4|w1,w2,w3) = λ4 * P_MLE(w4|w1,w2,w3) + λ3 * P_MLE(w4|w2,w3) + λ2 * P_MLE(w4|w3) + λ1 * P_MLE(w4)
where λ1+λ2+λ3+λ4=1, λi ≥ 0
best λs and example interpolated probabilities.
"""
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque
import random
import math

TRAIN_FILENAME = "train.txt"
VAL_FILENAME = "val.txt"
TEST_FILENAME = "test.txt"
MAX_N = 4

def stream_tokens(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                t = tok.strip()
                if t:
                    yield t

def count_ngrams(tokens, max_n=4):
    counts = {n: defaultdict(int) for n in range(1, max_n+1)}
    window = deque(maxlen=max_n-1)
    for tok in tokens:
        counts[1][(tok,)] += 1
        if max_n > 1:
            hist = list(window)
            hl = len(hist)
            for n in range(2, max_n+1):
                need = n-1
                if hl >= need:
                    gram = tuple(hist[-need:] + [tok])
                    counts[n][gram] += 1
        window.append(tok)
    return counts


def mle_prob(gram, counts_n, counts_prev):
    if len(gram) == 1:
        total = sum(counts_n.values())
        return counts_n.get(gram, 0) / total if total else 0.0
    hist = gram[:-1]
    denom = counts_prev.get(hist, 0)
    return counts_n.get(gram, 0) / denom if denom else 0.0


def deleted_interpolation_lambdas(train_counts, heldout_counts, vocab):
    # Grid search for best λs (coarse, then fine)
    best_lambdas = None
    best_logprob = float('-inf')
    # Coarse grid: step 0.1
    steps = [i/10 for i in range(11)]
    for l4 in steps:
        for l3 in steps:
            for l2 in steps:
                l1 = 1.0 - l4 - l3 - l2
                if l1 < 0 or l1 > 1:
                    continue
                lambdas = (l1, l2, l3, l4)
                logprob = 0.0
                total = 0
                for gram, c in heldout_counts[4].items():
                    w1, w2, w3, w4 = gram
                    p = (
                        l4 * mle_prob(gram, train_counts[4], train_counts[3]) +
                        l3 * mle_prob(gram[1:], train_counts[3], train_counts[2]) +
                        l2 * mle_prob(gram[2:], train_counts[2], train_counts[1]) +
                        l1 * mle_prob((w4,), train_counts[1], {})
                    )
                    if p > 0:
                        logprob += c * math.log(p)
                        total += c
                if logprob > best_logprob:
                    best_logprob = logprob
                    best_lambdas = lambdas
    return best_lambdas


def interpolated_prob(gram, counts, lambdas):
    w1, w2, w3, w4 = gram
    l1, l2, l3, l4 = lambdas
    p = (
        l4 * mle_prob(gram, counts[4], counts[3]) +
        l3 * mle_prob(gram[1:], counts[3], counts[2]) +
        l2 * mle_prob(gram[2:], counts[2], counts[1]) +
        l1 * mle_prob((w4,), counts[1], {})
    )
    return p

train_path = Path(TRAIN_FILENAME)
val_path = Path(VAL_FILENAME)
test_path = Path(TEST_FILENAME)
print(f"Reading train tokens from: {train_path}")
train_tokens = list(stream_tokens(train_path))
print(f"Reading val tokens from: {val_path}")
val_tokens = list(stream_tokens(val_path))
print(f"Reading test tokens from: {test_path}")
test_tokens = list(stream_tokens(test_path))
vocab = set(train_tokens)
train_counts = count_ngrams(train_tokens, max_n=4)
val_counts = count_ngrams(val_tokens, max_n=4)
print("Estimating best lambdas (deleted interpolation)...")
best_lambdas = deleted_interpolation_lambdas(train_counts, val_counts, vocab)
print(f"Best lambdas: λ1={best_lambdas[0]:.3f}, λ2={best_lambdas[1]:.3f}, λ3={best_lambdas[2]:.3f}, λ4={best_lambdas[3]:.3f}")
# Example: print top 5 quadrigrams and their interpolated probabilities on test set
print("\nTop 5 quadrigrams from test set and their interpolated probabilities:")
test_quad_counts = count_ngrams(test_tokens, max_n=4)[4]
rows = sorted(test_quad_counts.items(), key=lambda kv: -kv[1])[:5]
for gram, c in rows:
    p = interpolated_prob(gram, train_counts, best_lambdas)
    print(f"{' '.join(gram):<50} count={c:<5} P={p:.6g}")