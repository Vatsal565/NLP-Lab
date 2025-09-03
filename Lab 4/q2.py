"""
Lab 4 - Q2: Smoothing Techniques for N‑gram Models

Implements three smoothing variants for n-grams (n=2..4) over the Gujarati token corpus:
  a. Add-One (Laplace) Smoothing
	   P_add1(w_n | h) = (c(h, w_n) + 1) / (c(h) + V)
  b. Add-K Smoothing (generalized Laplace, K configurable)
	   P_addK(w_n | h) = (c(h, w_n) + K) / (c(h) + K * V)
  c. Add Token Type Smoothing (NOT guaranteed to form a probability distribution)
	   token_type_score(h, w_n) = c(h, w_n) + unique_char_count(w_n)

	   Rationale: Adds a pseudo-count proportional to lexical diversity of the predicted token.
	   Because we do not simultaneously adjust the denominator across vocabulary, these scores
	   are for ranking only (they will not sum to 1 over w_n).

Outputs:
  - bigrams_smoothing.tsv
  - trigrams_smoothing.tsv
  - quadragrams_smoothing.tsv

Each contains columns:
  w1 .. wn, count, mle_p, add1_p, addK_p, token_type_score

Process:
  * Stream tokens from indiccorp_gu_words.txt (no full list retained) for memory efficiency.
  * Maintain counts for n=1..4.
  * After counting, compute probabilities for n>=2.

Config knobs near top: INPUT_FILENAME, MAX_N, ADD_K, MAX_UNIQUE_PER_ORDER (optional pruning).

Note: Pruning (if enabled) may drop some rare higher-order n-grams (count==1) to save memory.
"""

from __future__ import annotations

from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque

# ---------------- Configuration ---------------- #
INPUT_FILENAME = "indiccorp_gu_words.txt"
MAX_N = 4              # up to quadragram
ADD_K = 0.5            # K for Add-K smoothing (change as desired)
TOP_PRINT = 8          # small preview in console
MAX_UNIQUE_PER_ORDER = None  # e.g., 600000 to cap memory; None disables pruning


# ---------------- File Location ---------------- #
def find_input_file() -> Path:
	here = Path(__file__).resolve().parent
	candidates = [
		here / INPUT_FILENAME,
		here.parent / INPUT_FILENAME,
		here.parent / "Lab 1" / INPUT_FILENAME,
		Path.cwd() / INPUT_FILENAME,
	]
	for p in candidates:
		if p.is_file():
			return p
	raise FileNotFoundError("Could not locate input file. Checked:\n" + "\n".join(str(c) for c in candidates))


# ---------------- Streaming Tokens ---------------- #
def stream_tokens(path: Path):
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			for tok in line.strip().split():
				t = tok.strip()
				if t:
					yield t


# ---------------- Counting ---------------- #
def prune_if_needed(counts: Dict[Tuple[str, ...], int]):
	if MAX_UNIQUE_PER_ORDER is None:
		return
	if len(counts) <= MAX_UNIQUE_PER_ORDER:
		return
	# remove count==1 entries
	to_drop = [k for k, v in counts.items() if v == 1]
	for k in to_drop:
		del counts[k]


# ---------------- Probability Helpers ---------------- #
def mle_conditional(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int]) -> float:
	hist = ngram[:-1]
	denom = counts_prev.get(hist, 0)
	return counts_n.get(ngram, 0) / denom if denom else 0.0


def add_one_conditional(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int], vocab_size: int) -> float:
	hist = ngram[:-1]
	denom = counts_prev.get(hist, 0)
	return (counts_n.get(ngram, 0) + 1) / (denom + vocab_size) if denom or vocab_size else 0.0


def add_k_conditional(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int], vocab_size: int, k: float) -> float:
	hist = ngram[:-1]
	denom = counts_prev.get(hist, 0)
	return (counts_n.get(ngram, 0) + k) / (denom + k * vocab_size) if denom or vocab_size else 0.0


def token_type_score(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int]) -> float:
	# Add number of unique characters in predicted token to raw count (NOT normalized)
	predicted = ngram[-1]
	return counts_n.get(ngram, 0) + len(set(predicted))


# ---------------- Output ---------------- #
def write_smoothed(n: int,
				   counts_n: Dict[Tuple[str, ...], int],
				   counts_prev: Dict[Tuple[str, ...], int],
				   vocab_size: int,
				   out_path: Path):
	rows = sorted(counts_n.items(), key=lambda kv: (-kv[1], kv[0]))
	header = [f"w{i+1}" for i in range(n)] + ["count", "mle_p", "add1_p", f"add{ADD_K}_p", "token_type_score"]
	with out_path.open("w", encoding="utf-8") as f:
		f.write("\t".join(header) + "\n")
		for gram, c in rows:
			mle_p = mle_conditional(gram, counts_n, counts_prev)
			add1_p = add_one_conditional(gram, counts_n, counts_prev, vocab_size)
			addk_p = add_k_conditional(gram, counts_n, counts_prev, vocab_size, ADD_K)
			tts = token_type_score(gram, counts_n)
			f.write("\t".join(list(gram) + [
				str(c),
				f"{mle_p:.8f}",
				f"{add1_p:.8f}",
				f"{addk_p:.8f}",
				f"{tts:.4f}",
			]) + "\n")


def print_preview(n: int, counts_n: Dict[Tuple[str, ...], int]):
	rows = sorted(counts_n.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_PRINT]
	print(f"Top {len(rows)} {n}-grams (by raw count):")
	for gram, c in rows:
		print(f"  {' '.join(gram):<60} {c}")
	print()


# ---------------- Main ---------------- #
def main():
	inp = find_input_file()
	print(f"Streaming tokens from: {inp}")

	counts: Dict[int, Dict[Tuple[str, ...], int]] = {i: defaultdict(int) for i in range(1, MAX_N + 1)}
	total_tokens = 0
	vocab = set()
	window: Deque[str] = deque(maxlen=MAX_N - 1)

	for tok in stream_tokens(inp):
		total_tokens += 1
		vocab.add(tok)
		counts[1][(tok,)] += 1
		if MAX_N > 1:
			hist = list(window)
			hl = len(hist)
			for n in range(2, MAX_N + 1):
				needed = n - 1
				if hl >= needed:
					gram = tuple(hist[-needed:] + [tok])
					counts[n][gram] += 1
					prune_if_needed(counts[n])
		window.append(tok)

	vocab_size = len(vocab)
	print(f"Total tokens: {total_tokens}; Vocabulary size: {vocab_size}")
	for n in range(1, MAX_N + 1):
		print(f"Unique {n}-grams: {len(counts[n])}")

	out_dir = Path(__file__).parent
	file_map = {2: "bigrams_smoothing.tsv", 3: "trigrams_smoothing.tsv", 4: "quadragrams_smoothing.tsv"}
	for n in range(2, MAX_N + 1):
		write_smoothed(n, counts[n], counts[n - 1], vocab_size, out_dir / file_map[n])
		print_preview(n, counts[n])

	print("Done. Files written:")
	for n in range(2, MAX_N + 1):
		print(f"  {file_map[n]}")
	print("Add-One & Add-K are proper probability distributions (sum to 1 per history). Token-type score is not.")


if __name__ == "__main__":
	main()
