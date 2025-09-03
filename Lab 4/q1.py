"""
Lab 4 - Simple N‑gram Models (Unigram, Bigram, Trigram, Quadragram)

Requirement adjustments:
 - Memory-friendly: stream tokens instead of loading whole list; maintain rolling window.
 - Outputs: unigrams.tsv, bigrams.tsv, trigrams.tsv, quadragrams.tsv
   * Unigrams: token \t count \t p(token)
   * Higher n: w1..wn \t count \t p(last|history)
 - Prints top 10 most frequent n-grams for each order.
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, Tuple, Deque

MAX_N = 4
TOP_PRINT = 10
MAX_UNIQUE_PER_ORDER = None  # e.g., 500000 to cap memory

def stream_tokens(path: Path):
	with path.open("r", encoding="utf-8", errors="ignore") as f:
		for line in f:
			for tok in line.strip().split():
				t = tok.strip()
				if t:
					yield t

def unigram_prob(count: int, total: int) -> float:
	return count / total if total else 0.0

def conditional_prob(ngram: Tuple[str, ...], counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int]) -> float:
	history = ngram[:-1]
	denom = counts_prev.get(history, 0) # frequency of history
	return counts_n.get(ngram, 0) / denom if denom else 0.0 # P(w_n | history)

def write_unigrams(counts: Dict[Tuple[str, ...], int], total: int, out_path: Path):
	rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
	with out_path.open("w", encoding="utf-8") as f:
		f.write("token\tcount\tp\n")
		for (tok,), c in rows:
			f.write(f"{tok}\t{c}\t{unigram_prob(c, total):.8f}\n")


def write_higher(n: int, counts_n: Dict[Tuple[str, ...], int], counts_prev: Dict[Tuple[str, ...], int], out_path: Path):
	rows = sorted(counts_n.items(), key=lambda kv: (-kv[1], kv[0]))
	header = [f"w{i+1}" for i in range(n)] + ["count", "p_cond"]
	with out_path.open("w", encoding="utf-8") as f:
		f.write("\t".join(header) + "\n")
		for gram, c in rows:
			p = conditional_prob(gram, counts_n, counts_prev)
			f.write("\t".join(list(gram) + [str(c), f"{p:.8f}"]) + "\n")


def print_top(counts: Dict[Tuple[str, ...], int], n: int):
	rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TOP_PRINT]
	print(f"Top {len(rows)} {n}-grams:")
	for gram, c in rows:
		print(f"  {' '.join(gram):<60} {c}")
	print()

inp = Path("indiccorp_gu_words.txt")
print(f"Streaming tokens from: {inp}")

counts: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, MAX_N + 1)}
total_tokens = 0
vocab = set()
window: Deque[str] = deque(maxlen=MAX_N - 1)

for tok in stream_tokens(inp):
	total_tokens += 1
	vocab.add(tok)
	counts[1][(tok,)] += 1
	if MAX_N > 1:
		hist = list(window)
		hl = len(hist) # ketli lambi history available che te
		for n in range(2, MAX_N + 1):
			need = n - 1 
			if hl >= need:
				gram = tuple(hist[-need:] + [tok])
				counts[n][gram] += 1
				
	window.append(tok)

print(f"Total tokens: {total_tokens}; Vocabulary size: {len(vocab)}")
for n in range(1, MAX_N + 1):
	print(f"Unique {n}-grams: {len(counts[n])}")

out_dir = Path(__file__).parent
write_unigrams(counts[1], total_tokens, out_dir / "unigrams.tsv")
print_top(counts[1], 1)

file_names = {2: "bigrams.tsv", 3: "trigrams.tsv", 4: "quadragrams.tsv"}
for n in range(2, MAX_N + 1):
	write_higher(n, counts[n], counts[n - 1], out_dir / file_names[n])
	print_top(counts[n], n)