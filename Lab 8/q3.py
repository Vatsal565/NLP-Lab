import math
import re
from collections import Counter, defaultdict
import pandas as pd

K = 0.3
EPS = 1e-12
BINARY_FEATURES = ("has_url", "has_number", "has_punct")

def tokenizer(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

def preprocess(text):
    # improved preprocessing, handles unicode words better
    tokens = [
        tok.lower()
        for tok in tokenizer(text)
        if tok.strip() and re.search(r"\w", tok, re.UNICODE)
    ]
    return tokens

def binary_flags(text):
    return {
        "has_url": int(bool(re.search(r"(https?://|www\.)", text, re.IGNORECASE))),
        "has_number": int(bool(re.search(r"\d", text))),
        "has_punct": int(bool(re.search(r"[^\w\s]", text, re.UNICODE))),
    }

def bigram_counter(tokens):
    return Counter(" ".join(pair) for pair in zip(tokens, tokens[1:]))

df = pd.read_csv("data.csv")
processed_lines = []
class_doc_counts = Counter()
class_binary_counts = defaultdict(Counter)
class_bigram_counts = defaultdict(Counter)
vocabulary = set()

for text, label in zip(df["Sentence"], df["Label"]):
    tokens = preprocess(text)
    processed_lines.append(" ".join(tokens))
    bigrams = bigram_counter(tokens)
    vocabulary.update(bigrams.keys())
    class_bigram_counts[label].update(bigrams)
    class_doc_counts[label] += 1
    flags = binary_flags(text)
    for feat, value in flags.items():
        if value:
            class_binary_counts[label][feat] += 1

with open("q2_preprocessed_sentences.txt", "w", encoding="utf-8") as fout:
    fout.write("\n".join(processed_lines))

total_docs = sum(class_doc_counts.values())
priors = {label: class_doc_counts[label] / total_docs for label in class_doc_counts}

# probs with add k smoothing
binary_probs = defaultdict(dict)
for label in class_doc_counts:
    docs = class_doc_counts[label]
    denom = docs + 2 * K
    for feat in BINARY_FEATURES:
        hits = class_binary_counts[label][feat]
        binary_probs[label][feat] = {
            "present": (hits + K) / denom,
            "absent": (docs - hits + K) / denom,
        }

vocab_size = max(len(vocabulary), 1)
bigram_probs = {}
for label, counts in class_bigram_counts.items():
    total_bigram = sum(counts.values())
    denom = total_bigram + K * vocab_size
    probs = {bg: (cnt + K) / denom for bg, cnt in counts.items()}
    bigram_probs[label] = {"known": probs, "unseen": K / denom}

print("Class priors:")
for label, prior in priors.items():
    print(f"  {label}: {prior:.6f}")

print("\nBinary feature probabilities:")
for label in sorted(binary_probs):
    print(f"  Class {label}:")
    for feat in BINARY_FEATURES:
        vals = binary_probs[label][feat]
        print(f"    {feat}: present={vals['present']:.6f}, absent={vals['absent']:.6f}")

print("\nBigram probabilities:")
for label in sorted(bigram_probs):
    stats = bigram_probs[label]
    for bg in sorted(stats["known"]):
        print(f"  {label} -> '{bg}': {stats['known'][bg]:.6f}")
    print(f"  {label} -> <unseen>: {stats['unseen']:.6f}")

test_sentence = "You will get an exclusive offer in the meeting!"
test_tokens = preprocess(test_sentence)
test_bigrams = bigram_counter(test_tokens)
test_flags = binary_flags(test_sentence)

def predict(text, priors, binary_probs, bigram_probs):
    tokens = preprocess(text)
    bigrams = bigram_counter(tokens)
    flags = binary_flags(text)

    raw_scores = {}
    for label in priors:
        log_p = math.log(priors[label] + EPS)
        for feat in BINARY_FEATURES:
            key = "present" if flags[feat] else "absent"
            log_p += math.log(binary_probs[label][feat][key] + EPS)
        stats = bigram_probs[label]
        for bg, cnt in bigrams.items():
            log_p += cnt * math.log(stats["known"].get(bg, stats["unseen"]) + EPS)
        raw_scores[label] = log_p

    predicted_label = max(raw_scores, key=lambda k: raw_scores[k])
    return predicted_label, raw_scores

print("\nBinary feature probabilities:")
for label, feats in binary_probs.items():
    print(f"  Class {label}:")
    for feat, vals in feats.items():
        print(
            f"    {feat}: present={vals['present']:.6f}, absent={vals['absent']:.6f}"
        )

print("\nBigram probabilities:")
for label, stats in bigram_probs.items():
    print(f"  Class {label}:")
    for bg, prob in sorted(stats["known"].items()):
        print(f"    '{bg}': {prob:.6f}")
    print(f"    <unseen>: {stats['unseen']:.6f}")

test_sentence = "You will get an exclusive offer in the meeting!"
predicted_label, raw_scores = predict(test_sentence, priors, binary_probs, bigram_probs)
print("\nScores for test sentence:")
for label, score in raw_scores.items():
    print(f"  {label}: {score:.6f}")
print(f"\nPredicted label: {predicted_label}")