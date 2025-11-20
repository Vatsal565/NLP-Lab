# katz backoff for quadrigram model
# alpha(h) = (1 - sum_{c(w,h)>k} P_katz(w|h)) / (1 - sum_{c(w,h)>0} P_katz_lower(w|h'))
# dw, h = ((r + 1) * Nr+1) / (Nr * r)
# Pkatz(W | h) = alpha * Pkatz(W | Wi-1, Wi-2, Wi-3) where c(w | h) <= k
# Pkatz(W | h) = dw, h * Pmle(w | h) where c(w | h) > k
from pathlib import Path
from collections import defaultdict, deque, Counter

TRAIN_FILENAME = "temp_train.txt"
MAX_N = 4
k = 1  # threshold for discounting (can be tuned)

def stream_tokens(path: Path):
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
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

def discount(c, Nc, Nc1):
	# dw, h = ((r + 1) * Nr+1) / (Nr * r)
	r = c
	if r > 0 and Nc.get(r, 0) > 0 and Nc.get(r+1, 0) > 0:
		return (r+1) * Nc[r+1] / (Nc[r] * r)
	return 1.0

def get_Nc(counts_n):
	Nc = Counter(counts_n.values())
	return Nc

def pkatz(w, h, counts, Nc_dict, k):
	# h: tuple of 3 words (Wi-1, Wi-2, Wi-3)
	quad = h + (w,)
	c = counts[4].get(quad, 0)
	Nc = Nc_dict[4]
	if c > k:
		dw_h = discount(c, Nc, Nc)
		return dw_h * mle_prob(quad, counts[4], counts[3])
	else:
		# Backoff
		alpha = compute_alpha(h, counts, Nc_dict, k)
		return alpha * pkatz_lower(w, h[1:], counts, Nc_dict, k)

def pkatz_lower(w, h, counts, Nc_dict, k):
	# h: tuple of 2 words (Wi-2, Wi-3)
	tri = h + (w,)
	c = counts[3].get(tri, 0)
	Nc = Nc_dict[3]
	if c > k:
		dw_h = discount(c, Nc, Nc)
		return dw_h * mle_prob(tri, counts[3], counts[2])
	else:
		alpha = compute_alpha(h, counts, Nc_dict, k, order=3)
		return alpha * pkatz_lower2(w, h[1:], counts, Nc_dict, k)

def pkatz_lower2(w, h, counts, Nc_dict, k):
	# h: tuple of 1 word (Wi-3)
	bi = h + (w,)
	c = counts[2].get(bi, 0)
	Nc = Nc_dict[2]
	if c > k:
		dw_h = discount(c, Nc, Nc)
		return dw_h * mle_prob(bi, counts[2], counts[1])
	else:
		alpha = compute_alpha(h, counts, Nc_dict, k, order=2)
		return alpha * mle_prob((w,), counts[1], {})

def compute_alpha(h, counts, Nc_dict, k, order=4):
	# h: tuple of order-1 words
	# alpha = (1 - sigma(PKatz(Wi | h, ...))) / (1 - sigma(PKatz(Wi | lower history)))
	V = len(counts[1])
	numer = 1.0
	denom = 1.0
	# For all w in vocab
	for w in counts[1]:
		wtok = w[0]
		if order == 4:
			quad = h + (wtok,)
			c = counts[4].get(quad, 0)
			Nc = Nc_dict[4]
			if c > k:
				numer -= discount(c, Nc, Nc) * mle_prob(quad, counts[4], counts[3])
		elif order == 3:
			tri = h + (wtok,)
			c = counts[3].get(tri, 0)
			Nc = Nc_dict[3]
			if c > k:
				numer -= discount(c, Nc, Nc) * mle_prob(tri, counts[3], counts[2])
		elif order == 2:
			bi = h + (wtok,)
			c = counts[2].get(bi, 0)
			Nc = Nc_dict[2]
			if c > k:
				numer -= discount(c, Nc, Nc) * mle_prob(bi, counts[2], counts[1])
	# Denominator: sum of lower order
	for w in counts[1]:
		wtok = w[0]
		if order == 4:
			denom -= pkatz_lower(wtok, h[1:], counts, Nc_dict, k)
		elif order == 3:
			denom -= pkatz_lower2(wtok, h[1:], counts, Nc_dict, k)
		elif order == 2:
			denom -= mle_prob((wtok,), counts[1], {})
	if denom == 0:
		return 1.0
	return numer / denom

train_path = Path(__file__).parent / TRAIN_FILENAME
tokens = list(stream_tokens(train_path))
counts = count_ngrams(tokens, max_n=4)
Nc_dict = {n: get_Nc(counts[n]) for n in range(1, 5)}

# Pick a random quadgram from training
quadgrams = list(counts[4].keys())
if quadgrams:
    h = quadgrams[0][:3]
    w = quadgrams[0][3]
    print("Performing Katz Backoff Probability Calculation for", h, w)
    pk = pkatz(w, h, counts, Nc_dict, k)
    print(f"P_Katz({w} | {h}) = {pk:.8f}")