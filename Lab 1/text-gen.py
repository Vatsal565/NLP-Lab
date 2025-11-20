from pathlib import Path
from collections import Counter, defaultdict, deque
import math
import heapq

TRAIN = Path("msd.txt")
OUT_DIR = Path(".")
MAX_N = 4
BEAM_SIZE = 20
BEAM_CANDIDATES = 50
MAX_LEN = 40

def build_counts(path, max_n=4):
    ngram_counts = {n: Counter() for n in range(1, max_n+1)}
    context_counts = {n: Counter() for n in range(2, max_n+1)}
    vocab = set()

    with open(path, encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue

            sent = ["<s>"] * (max_n-1) + toks + ["</s>"]

            for n in range(1, max_n+1):
                for i in range(len(sent)-n+1):
                    gram = tuple(sent[i:i+n])
                    ngram_counts[n][gram] += 1
                    if n >= 2:
                        ctx = tuple(sent[i:i+n-1])
                        context_counts[n][ctx] += 1

            vocab.update(toks)

    vocab.update({"<s>", "</s>"})
    return ngram_counts, context_counts, sorted(vocab)

ngram_counts, context_counts, VOCAB = build_counts(TRAIN, MAX_N)
TOTAL_UNIGRAMS = sum(ngram_counts[1].values())

def mle_prob(next_word, context):
    """
    Highest-order MLE with backoff.
    context is a tuple of tokens.
    """

    # max usable context = n-1 tokens
    for order in range(min(len(context), MAX_N-1), -1, -1):

        if order == 0:
            # unigram
            count = ngram_counts[1].get((next_word,), 0)
            return count / TOTAL_UNIGRAMS if TOTAL_UNIGRAMS else 0.0

        ctx = tuple(context[-order:])
        denom = context_counts[order+1].get(ctx, 0)
        if denom == 0:
            continue

        num = ngram_counts[order+1].get(ctx + (next_word,), 0)
        return num / denom

    return 0.0

def top_k_candidates(context, k=50):
    heap = []

    for w in VOCAB:
        if w == "<s>":     # do not generate <s> again
            continue

        p = mle_prob(w, context)
        if p <= 0:
            continue

        if len(heap) < k:
            heapq.heappush(heap, (p, w))
        else:
            if p > heap[0][0]:
                heapq.heapreplace(heap, (p, w))

    # heap holds (prob, word) → return sorted descending
    return sorted([(w, p) for p, w in heap], key=lambda x: -x[1])

def generate_greedy(n):
    ctx = deque(["<s>"] * (n-1), maxlen=n-1)
    out = []

    for _ in range(MAX_LEN):
        best_w, best_p = None, -1

        for w in VOCAB:
            if w == "<s>":
                continue

            p = mle_prob(w, tuple(ctx))
            if p > best_p:
                best_p = p
                best_w = w

        if best_w is None:
            break

        out.append(best_w)
        if best_w == "</s>":
            break

        ctx.append(best_w)

    if out and out[-1] == "</s>":
        out = out[:-1]

    return " ".join(out)

def generate_beam_unigram():
    # Beam search for unigram model = pick top tokens by unigram freq
    top_words = ngram_counts[1].most_common(BEAM_SIZE)
    # remove </s> and <s>
    top_words = [w for w, c in top_words if w not in ("<s>", "</s>")]
    if not top_words:
        return ""
    return " ".join(top_words[1])

def generate_beam(n, beam_size=BEAM_SIZE):
    if n == 1:
        return generate_beam_unigram()
    # beam entry = (logP, tokens, context, finished_flag)
    start_ctx = deque(["<s>"] * (n-1), maxlen=n-1)
    beams = [(0.0, [], start_ctx.copy(), False)]
    completed = []

    for _ in range(MAX_LEN):

        new_beams = []

        for logp, toks, ctx, done in beams:

            if done:
                new_beams.append((logp, toks, ctx, True))
                continue

            # expand
            candidates = top_k_candidates(tuple(ctx), BEAM_CANDIDATES)
            if not candidates:
                new_beams.append((logp, toks, ctx, True))
                continue

            for w, p in candidates:
                new_ctx = ctx.copy()
                new_ctx.append(w)
                new_toks = toks + [w]
                new_logp = logp + math.log(p)
                finished = (w == "</s>")
                new_beams.append((new_logp, new_toks, new_ctx, finished))

        if not new_beams:
            break

        # prune to beam_size
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        # collect finished beams
        for lp, tks, c, f in beams:
            if f and tks not in completed:
                completed.append((lp, tks))

        if len(completed) >= beam_size:
            break

    # ranking final sequences
    final_results = completed if completed else [(lp, toks) for lp, toks, _, _ in beams]
    final_results.sort(key=lambda x: x[0], reverse=True)

    # return best sentence
    best = final_results[0][1]
    if best and best[-1] == "</s>":
        best = best[:-1]
    return " ".join(best)

def write_sentences(model_n):
    gfile = OUT_DIR / f"{model_n}gram_greedy.txt"
    bfile = OUT_DIR / f"{model_n}gram_beam.txt"

    with open(gfile, "w", encoding="utf-8") as gf, open(bfile, "w", encoding="utf-8") as bf:
        for _ in range(100):
            gf.write(generate_greedy(model_n) + "\n")
            bf.write(generate_beam(model_n, BEAM_SIZE) + "\n")

for n in range(1, MAX_N+1):
    write_sentences(n)
