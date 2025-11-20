# kneser-ney smoothing
# p(Wi | Wi-1, Wi-2, Wi-3) = max(c(Wi-3, Wi-2, Wi-1, Wi) - d, 0) / c(Wi-3, Wi-2, Wi-1) + lambda(Wi-3, Wi-2, Wi-1) * p(Wi)
# lambda(Wi-1, Wi-2, Wi-3) = (d / c(Wi-3, Wi-2, Wi-1)) * unique quadrigrams starting with Wi-3, Wi-2, Wi-1 
# P(Wi) = no. of unigram ending with Wi / total no. of unique quadrigrams
"""
Kneser-Ney smoothing implementation for n=1..4 (quadrigram model).
Formulas:
 p_kn(w|h) = max(c(h w) - d, 0)/c(h) + lambda(h) * p_kn_lower(w|h')
 lambda(h) = (d / c(h)) * |{w: c(h w) > 0}|
 Base unigram p_kn_1(w) = continuation_count(w) / total_continuation
"""
from pathlib import Path
from collections import defaultdict, deque, Counter

TRAIN_FILENAME = "temp_train.txt"
MAX_N = 4
DISCOUNT = 0.75  

def stream_tokens(path: Path):
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			for tok in line.strip().split():
				t = tok.strip()
				if t:
					yield t

def count_ngrams(tokens, max_n=4):
	counts = {n: defaultdict(int) for n in range(1, max_n + 1)}
	window = deque(maxlen=max_n - 1)
	for tok in tokens:
		counts[1][(tok,)] += 1
		if max_n > 1:
			hist = list(window)
			hl = len(hist)
			for n in range(2, max_n + 1):
				need = n - 1
				if hl >= need:
					gram = tuple(hist[-need:] + [tok])
					counts[n][gram] += 1
		window.append(tok)
	return counts


def compute_continuation_counts(counts):
	# continuation counts for unigram: number of unique left contexts for w (from bigrams)
	cont_uni = Counter()
	for (u, v), c in counts[2].items():
		cont_uni[v] += 1

	# For higher orders, unique continuation counts per history h: |{w: c(h w) > 0}|
	unique_cont = {n: {} for n in range(2, MAX_N + 1)}
	for n in range(2, MAX_N + 1):
		uniq = defaultdict(int)
		for gram in counts[n].keys():
			h = gram[:-1]
			uniq[h] += 1
		unique_cont[n] = uniq

	total_bigram_types = len(counts[2])
	return cont_uni, total_bigram_types, unique_cont


def p_kn_recursive(w, h, counts, cont_uni, total_bigram_types, unique_cont, d=DISCOUNT):
	n = len(h) + 1
	if n == 1:
		# unigram base: continuation probability
		return cont_uni.get(w, 0) / total_bigram_types if total_bigram_types > 0 else 0.0

	# c(h) and c(h w)
	c_h = counts[n - 1].get(h, 0)
	gram = h + (w,)
	c_hw = counts[n].get(gram, 0)

	# first term
	term1 = 0.0
	if c_h > 0:
		term1 = max(c_hw - d, 0.0) / c_h

	# lambda(h)
	lambda_h = 0.0
	if c_h > 0:
		N_cont_h = unique_cont[n].get(h, 0)
		lambda_h = (d / c_h) * N_cont_h

	# lower-order history: drop leftmost word
	lower_h = h[1:]
	p_lower = p_kn_recursive(w, lower_h, counts, cont_uni, total_bigram_types, unique_cont, d)
	return term1 + lambda_h * p_lower

train_path = Path(TRAIN_FILENAME)
tokens = list(stream_tokens(train_path))
counts = count_ngrams(tokens, max_n=MAX_N)
cont_uni, total_bigram_types, unique_cont = compute_continuation_counts(counts)

quad_counts = counts[4]
top_quads = sorted(quad_counts.items(), key=lambda kv: -kv[1])[:10]
print("Top 10 quadrigrams with Kneser-Ney probabilities:")
for gram, c in top_quads:
    h = gram[:-1]
    w = gram[-1]
    p = p_kn_recursive(w, h, counts, cont_uni, total_bigram_types, unique_cont)
    print(f"{' '.join(gram):<40} count={c:<6} p_kn={p:.4f}")

