from pathlib import Path
from collections import defaultdict, deque, Counter
from typing import Dict, Tuple, Deque

INPUT_FILENAME = "train.txt"
MAX_N = 4
MAX_UNIQUE_PER_ORDER = None  

def stream_tokens(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                t = tok.strip()
                if t:
                    yield t

def good_turing_probs(counts: Dict[Tuple[str, ...], int], n: int, vocab_size: int) -> Tuple[Dict[Tuple[str, ...], float], float]:
    # Compute Nc: Nc[c] = number of n-grams with count c
    Nc = Counter(counts.values())
    N = sum(counts.values())
    N1 = Nc[1]
    if n == 1:
        U = len(counts)
        num_unseen = vocab_size - U
        P_unseen = N1 / (N * num_unseen) if num_unseen > 0 else 0.0
    else:
        num_unseen = vocab_size ** n - len(counts)
        P_unseen = N1 / (N * num_unseen) if num_unseen > 0 else 0.0
    probs = {}
    cstar_table = {}
    max_c = max(Nc) if Nc else 0
    for c in range(0, max_c + 1):
        Nc1 = Nc.get(c + 1, 0)
        Nc_c = Nc.get(c, 0)
        if c == 0:
            cstar = N1 / num_unseen if num_unseen > 0 else 0.0
        elif Nc1 > 0 and Nc_c > 0:
            cstar = (c + 1) * Nc1 / Nc_c
        else:
            cstar = c
        cstar_table[c] = (Nc_c, cstar)
    for gram, c in counts.items():
        Nc1 = Nc.get(c + 1, 0)
        if Nc1 > 0:
            c_star = (c + 1) * Nc1 / Nc[c]
            p = c_star / N
        else:
            p = c / N
        probs[gram] = p
    return probs, P_unseen, cstar_table


def write_gt(n: int, counts: Dict[Tuple[str, ...], int], probs: Dict[Tuple[str, ...], float], out_path: Path):
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    header = [f"w{i+1}" for i in range(n)] + ["count", "gt_prob"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for gram, c in rows:
            p = probs[gram]
            f.write("\t".join(list(gram) + [str(c), f"{p:.8f}"]) + "\n")


inp = Path("indiccorp_gu_words.txt")
print(f"Streaming tokens from: {inp}")
counts: Dict[int, Dict[Tuple[str, ...], int]] = {n: defaultdict(int) for n in range(1, MAX_N + 1)}
vocab = set()
window: Deque[str] = deque(maxlen=MAX_N - 1)
for tok in stream_tokens(inp):
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
                
    window.append(tok)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")
name_map = {1: "unigrams_gt.tsv", 2: "bigrams_gt.tsv", 3: "trigrams_gt.tsv", 4: "quadragrams_gt.tsv"}
for n in range(1, MAX_N + 1):
    print(f"\n--- {n}-gram Model ---")
    probs, P_unseen, cstar_table = good_turing_probs(counts[n], n, vocab_size)
    print(f"Probability for unseen {n}-gram: {P_unseen:.8g}")
    write_gt(n, counts[n], probs, Path(__file__).parent / name_map[n])
    # Print top 100 frequency table: C, Nc, C*
    print(f"Top 100 frequency table for {n}-grams:")
    print(f"{'C':>6} {'Nc':>8} {'C*':>12}")
    # Always show C=0 row
    print(f"{0:6d} {cstar_table[0][0]:8d} {cstar_table[0][1]:12.4f}")
    # Then top 99 most common C>0
    freq_counts = sorted([(c, Nc) for c, (Nc, _) in cstar_table.items() if c > 0], key=lambda x: -x[1])[:99]
    for c, Nc_val in freq_counts:
        cstar = cstar_table[c][1]
        print(f"{c:6d} {Nc_val:8d} {cstar:12.4f}")