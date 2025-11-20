from pathlib import Path
from collections import Counter, defaultdict, deque
import math
import heapq

TRAIN = Path("temp_train.txt") 
OUT_DIR = Path(".")
MAX_N = 4
BEAM_SIZE = 20
BEAM_CANDIDATES = 50  # har ek beam step par ketla branches ne consider karvana
MAX_LEN = 40

def build_counts(path, max_n=4):
    ngram_counts = {n: Counter() for n in range(1, max_n+1)}
    context_counts = {n: Counter() for n in range(1, max_n+1)}  # context for n-grams (n>=2)
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
            for t in toks:
                vocab.add(t)
            vocab.add("<s>"); vocab.add("</s>")
    return ngram_counts, context_counts, sorted(vocab)

ngram_counts, context_counts, VOCAB = build_counts(TRAIN, MAX_N)
TOTAL_UNIGRAMS = sum(ngram_counts[1].values())

def mle_prob(next_word, context):
    # tries highest-order conditional, backs off to shorter contexts (MLE)
    # context is tuple of previous tokens, keep up to MAX_N-1
    for order in range(min(len(context), MAX_N-1), -1, -1):
        if order == 0:
            # unigram
            cnt = ngram_counts[1].get((next_word,), 0)
            if TOTAL_UNIGRAMS == 0: return 0.0
            return cnt / TOTAL_UNIGRAMS
        ctx = tuple(context[-order:])
        denom = context_counts[order+1].get(ctx, 0)
        if denom == 0:
            continue
        cnt = ngram_counts[order+1].get(ctx + (next_word,), 0)
        return cnt / denom
    return 0.0

def top_k_candidates(context, k=50):
    # return list of (word, prob) top k by mle_prob
    heap = []
    for w in VOCAB:
        p = mle_prob(w, context)
        if p > 0:
            if len(heap) < k:
                heapq.heappush(heap, (p, w))
            else:
                if p > heap[0][0]:
                    heapq.heapreplace(heap, (p, w))
    return sorted([(w, p) for p, w in heap], key=lambda x: -x[1])

def generate_greedy(n):
    out = []
    ctx = deque(["<s>"] * (n-1), maxlen=n-1)
    for _ in range(MAX_LEN):
        # pick argmax
        best_w, best_p = None, 0.0
        for w in VOCAB:
            p = mle_prob(w, tuple(ctx))
            if p > best_p:
                best_p = p; best_w = w
        if best_w is None:
            break
        out.append(best_w)
        if best_w == "</s>":
            break
        ctx.append(best_w)
    # strip end token
    if out and out[-1] == "</s>":
        out = out[:-1]
    return " ".join(out)

def generate_beam(n, beam_size=BEAM_SIZE):
    # beams: list of tuples (logprob, context deque, tokens list, finished)
    start_ctx = deque(["<s>"] * (n-1), maxlen=n-1)
    beams = [(0.0, start_ctx.copy(), [], False)]
    completed = []
    while beams and len(completed) < beam_size:
        new_beams = []
        for logp, ctx, toks, finished in beams:
            if finished or (toks and toks[-1] == "</s>"):
                completed.append((logp, toks))
                if len(completed) >= beam_size:
                    continue
                new_beams.append((logp, ctx, toks, True))
                continue
            # expand with top candidates
            cands = top_k_candidates(tuple(ctx), BEAM_CANDIDATES)
            if not cands:
                # force stop
                completed.append((logp, toks))
                continue
            for w, p in cands:
                if p <= 0: continue
                new_ctx = ctx.copy()
                new_ctx.append(w)
                new_toks = toks + [w]
                new_logp = logp + math.log(p)
                finished_flag = (w == "</s>")
                new_beams.append((new_logp, new_ctx, new_toks, finished_flag))
        if not new_beams:
            break
        # keep top beam_size beams by logprob
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]
    # collect completed sequences, if none use current beams
    results = completed if completed else [(lp, toks) for lp, _, toks, _ in beams]
    # sort and return tokens of top beam_size
    results.sort(key=lambda x: x[0], reverse=True)
    sentences = []
    for lp, toks in results[:beam_size]:
        sent = toks.copy()
        if sent and sent[-1] == "</s>":
            sent = sent[:-1]
        sentences.append(" ".join(sent))
    # ensure we return at least one
    return sentences[0] if sentences else ""

def write_sentences(model_n):
    gfile = OUT_DIR / f"{model_n}gram_greedy.txt"
    bfile = OUT_DIR / f"{model_n}gram_beam.txt"
    with open(gfile, "w", encoding="utf-8") as gf, open(bfile, "w", encoding="utf-8") as bf:
        for i in range(100):
            s_g = generate_greedy(model_n)
            s_b = generate_beam(model_n, BEAM_SIZE)
            gf.write(s_g + "\n")
            bf.write(s_b + "\n")

for n in range(1, MAX_N+1):
    write_sentences(n)