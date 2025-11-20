from pathlib import Path
from collections import defaultdict, deque, Counter

INPUT_FILE = "msd.txt"
MAX_N = 4

def stream_tokens(filename):
    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                t = tok.strip()
                if t:
                    yield t

def good_turing(counts, n, vocab_size):
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
        Nc[0] = num_unseen
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

def write_gt(n: int, counts, probs, out_path: Path):
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    header = [f"w{i+1}" for i in range(n)] + ["count", "gt_prob"]
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for gram, c in rows:
            p = probs[gram]
            f.write("\t".join(list(gram) + [str(c), f"{p:.8f}"]) + "\n")

AllCounts = defaultdict(lambda: defaultdict(int))
Vocab = set()
Window = deque(maxlen=MAX_N - 1)
OutputDir = Path("out")

# 1. Collect counts and build vocabulary
for token in stream_tokens(INPUT_FILE):
    Vocab.add(token)
    AllCounts[1][(token,)] += 1
    
    if MAX_N > 1:
        Hist = list(Window)
        for n_gram in range(2, MAX_N + 1):
            Need = n_gram - 1
            if len(Hist) >= Need:
                gram = tuple(Hist[-Need:] + [token])
                AllCounts[n_gram][gram] += 1
    Window.append(token)
    
VocabSize = len(Vocab)
print(f"Vocab size: {VocabSize}")

# 2. Compute GT probabilities and write output
NameMap = {1: "unigrams_gt.tsv", 2: "bigrams_gt.tsv", 3: "trigrams_gt.tsv", 4: "quadragrams_gt.tsv"}

for n_gram in range(1, MAX_N + 1):
    n_counts = AllCounts.get(n_gram)
    if not n_counts: continue

    Probs, P_unseen, CstarTable = good_turing(n_counts, n_gram, VocabSize)

    OutPath = OutputDir / NameMap.get(n_gram, f"{n_gram}grams_gt.tsv")
    write_gt(n_gram, n_counts, Probs, OutPath)

    print(f"\n--- {n_gram}-gram Model ---\nUnseen P: {P_unseen:.8g}\n{'C':>6} {'Nc':>8} {'C*':>12}")

    # C=0 row
    c0_Nc, c0_cstar = CstarTable[0]
    print(f"{0:6d} {c0_Nc:8d} {c0_cstar:12.4f}")

    # Top 99 C > 0
    freq_counts = sorted([(C, Nc) for C, (Nc, _) in CstarTable.items() if C > 0], key=lambda x: -x[1])[:99]
    for C, Nc in freq_counts:
        print(f"{C:6d} {Nc:8d} {CstarTable[C][1]:12.4f}")