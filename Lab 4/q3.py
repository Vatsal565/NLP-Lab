"""
Q3: Sentence Probability with Smoothed N-gram Models

Task:
  Use the smoothed language models (Add-One, Add-K, Token-Type score) to compute
  probabilities for each sentence in q3_data.txt using n = 2, 3, 4 (bigrams, trigrams, quadragrams).

Models:
  Add-One (Laplace):      P(w|h) = (c(h,w)+1)/(c(h)+V)
  Add-K (general K>0):    P(w|h) = (c(h,w)+K)/(c(h)+K*V)
  Token-Type score:       score(h,w) = c(h,w) + unique_char_count(w)  (NOT a probability)

Implementation notes:
  * Stream corpus tokens from indiccorp_gu_words.txt (no full token list held) to build counts for n=1..4.
  * Sentence tokens are taken directly from q3_data.txt (simple whitespace split after stripping the numeric prefix "N.").
  * We do NOT introduce <s> or </s> markers to keep consistent with training counts.
  * For an n-gram whose history never appeared (denom=0) we still apply smoothing denominator (c(h)=0):
		Add-One: (0+1)/(0+V)
		Add-K:   (0+K)/(0+K*V)
  * Quadragram probabilities back off implicitly when history length < n-1 by using the available preceding tokens (i.e. we only form n-grams once enough tokens observed). For first few tokens of a sentence under higher-order model we use progressively smaller n (unigram probability for the first token).
  * Log probabilities (base 10) are reported to avoid underflow. Also perplexity = 10^(-log10P / m) where m = number of conditional factors used.
  * Token-Type scores are summed (not multiplied) per sentence to give a comparable ranking feature; they do not form a probability distribution.

Outputs:
  sentence_probs.tsv with columns:
	 sent_id \t n \t tokens_used \t add1_log10P \t add1_perplexity \t addK_log10P \t addK_perplexity \t token_type_sum

Config knobs below: INPUT_FILENAME, SENTENCE_FILE, ADD_K, MAX_N, MAX_UNIQUE_PER_ORDER.
"""

from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque, List
import math
import re

######## Configuration ########
INPUT_FILENAME = "indiccorp_gu_words.txt"
SENTENCE_FILE = "q3_data.txt"
ADD_K = 0.5          # K value for Add-K smoothing
MAX_N = 4            # build up to quadragram counts
MAX_UNIQUE_PER_ORDER = None  # optional pruning cap; None disables
NGRAM_ORDERS = (2, 3, 4)      # which n values to evaluate for sentences


######## File Discovery ########
def find_file(name: str) -> Path:
	here = Path(__file__).resolve().parent
	candidates = [
		here / name,
		here.parent / name,
		here.parent / "Lab 1" / name,
		Path.cwd() / name,
	]
	for p in candidates:
		if p.is_file():
			return p
	raise FileNotFoundError(f"Could not locate {name}. Checked:\n" + "\n".join(str(c) for c in candidates))


######## Streaming Corpus Tokens ########
def stream_tokens(path: Path):
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			for tok in line.strip().split():
				t = tok.strip()
				if t:
					yield t


######## Counting ########
def prune_if_needed(counts: Dict[Tuple[str, ...], int]):
	if MAX_UNIQUE_PER_ORDER is None:
		return
	if len(counts) <= MAX_UNIQUE_PER_ORDER:
		return
	# remove singletons
	to_drop = [k for k, v in counts.items() if v == 1]
	for k in to_drop:
		del counts[k]


######## Probability Helpers ########
def add_one_prob(count_hw: int, count_h: int, V: int) -> float:
	return (count_hw + 1) / (count_h + V) if V else 0.0


def add_k_prob(count_hw: int, count_h: int, V: int, k: float) -> float:
	return (count_hw + k) / (count_h + k * V) if V else 0.0


def token_type_score(count_hw: int, word: str) -> float:
	return count_hw + len(set(word))  # pseudo count (not normalized)


######## Sentence Processing ########
def read_sentences(path: Path) -> List[Tuple[int, List[str]]]:
	sentences: List[Tuple[int, List[str]]] = []
	num_prefix = re.compile(r"^\s*(\d+)\.\s*")
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			m = num_prefix.match(line)
			sid = None
			if m:
				sid = int(m.group(1))
				line = line[m.end():]
			else:
				sid = len(sentences) + 1
			# simple whitespace tokenization
			toks = [t for t in line.split() if t]
			sentences.append((sid, toks))
	return sentences


def sentence_prob(tokens: List[str], n: int, counts: Dict[int, Dict[Tuple[str, ...], int]], vocab_size: int) -> Tuple[float, float, float]:
	"""Return (add1_log10P, addK_log10P, token_type_sum) for the sentence with model order n.
	We back off implicitly at sentence start (use shorter histories until enough tokens seen).
	log10P computed over the sequence of conditional factors actually formed.
	"""
	log10_add1 = 0.0
	log10_addK = 0.0
	token_type_sum = 0.0
	factors = 0
	history: List[str] = []
	for w in tokens:
		history.append(w)  # we will use history excluding current; adjust below
		# Determine effective history length (up to n-1 previous words)
		if len(history) == 1:  # first token; use unigram prob approximation via total count
			# Unigram counts
			count_w = counts[1].get((w,), 0)
			total_tokens = sum(counts[1].values())
			# Add-one on unigram: (c(w)+1)/(N+V)
			add1_p = (count_w + 1) / (total_tokens + vocab_size)
			addK_p = (count_w + ADD_K) / (total_tokens + ADD_K * vocab_size)
			token_type_sum += token_type_score(count_w, w)
		else:
			use_hist = history[:-1]
			# We only attempt to form n-gram if enough context else shrink order
			order = n
			while order > 1 and len(use_hist) < order - 1:
				order -= 1
			if order == 1:
				count_w = counts[1].get((w,), 0)
				total_tokens = sum(counts[1].values())
				add1_p = (count_w + 1) / (total_tokens + vocab_size)
				addK_p = (count_w + ADD_K) / (total_tokens + ADD_K * vocab_size)
				token_type_sum += token_type_score(count_w, w)
			else:
				hist_slice = tuple(use_hist[-(order - 1):])
				gram = hist_slice + (w,)
				count_hw = counts[order].get(gram, 0)
				count_h = counts[order - 1].get(hist_slice, 0)
				add1_p = add_one_prob(count_hw, count_h, vocab_size)
				addK_p = add_k_prob(count_hw, count_h, vocab_size, ADD_K)
				token_type_sum += token_type_score(count_hw, w)
		# Avoid log(0)
		if add1_p <= 0:
			add1_p = 1e-20
		if addK_p <= 0:
			addK_p = 1e-20
		log10_add1 += math.log10(add1_p)
		log10_addK += math.log10(addK_p)
		factors += 1
	return log10_add1, log10_addK, token_type_sum


######## Main ########
def main():
	corpus_path = find_file(INPUT_FILENAME)
	sent_path = find_file(SENTENCE_FILE)
	print(f"Building n-gram counts from: {corpus_path}")

	counts: Dict[int, Dict[Tuple[str, ...], int]] = {i: defaultdict(int) for i in range(1, MAX_N + 1)}
	vocab = set()
	window: Deque[str] = deque(maxlen=MAX_N - 1)
	total_tokens = 0
	for tok in stream_tokens(corpus_path):
		total_tokens += 1
		vocab.add(tok)
		counts[1][(tok,)] += 1
		if MAX_N > 1:
			hist = list(window)
			hl = len(hist)
			for n in range(2, MAX_N + 1):
				need = n - 1
				if hl >= need:
					gram = tuple(hist[-need:] + [tok])
					counts[n][gram] += 1
					prune_if_needed(counts[n])
		window.append(tok)
	vocab_size = len(vocab)
	print(f"Total tokens: {total_tokens}; Vocab size: {vocab_size}")

	sentences = read_sentences(sent_path)
	print(f"Loaded {len(sentences)} sentences from {SENTENCE_FILE}")

	out_path = Path(__file__).parent / "sentence_probs.tsv"
	with out_path.open("w", encoding="utf-8") as f:
		f.write("sent_id\tn\ttokens_used\tadd1_log10P\tadd1_perplexity\taddK_log10P\taddK_perplexity\ttoken_type_sum\n")
		for sid, toks in sentences:
			for n in NGRAM_ORDERS:
				log10_add1, log10_addK, tts = sentence_prob(toks, n, counts, vocab_size)
				m = len(toks)
				add1_perp = 10 ** (-log10_add1 / m) if m else 0.0
				addK_perp = 10 ** (-log10_addK / m) if m else 0.0
				f.write(
					f"{sid}\t{n}\t{m}\t{log10_add1:.6f}\t{add1_perp:.4f}\t{log10_addK:.6f}\t{addK_perp:.4f}\t{tts:.2f}\n"
				)
	print(f"Wrote sentence probabilities to {out_path}")


if __name__ == "__main__":
	main()
