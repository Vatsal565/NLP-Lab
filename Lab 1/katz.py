from pathlib import Path
from collections import defaultdict, deque, Counter

TRAIN_FILENAME = "msd.txt"
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
                need = n - 1
                if hl >= need:
                    gram = tuple(hist[-need:] + [tok])
                    counts[n][gram] += 1
        window.append(tok)
    return counts

def mle_prob(gram, counts_n, counts_prev):
    # If unigram
    if len(gram) == 1:
        total = sum(counts_n.values())
        return counts_n.get(gram, 0) / total if total else 0.0
    # For n-gram (len>1), denominator is count(history)
    hist = gram[:-1]
    denom = counts_prev.get(hist, 0)
    return counts_n.get(gram, 0) / denom if denom else 0.0

def discount(c, Nc):
    """
    Discount formula:
      d_{r} = ((r + 1) * N_{r+1}) / (N_r * r)
    where Nc is a Counter mapping frequency r -> number of n-grams with frequency r
    """
    r = c
    if r > 0 and Nc.get(r, 0) > 0 and Nc.get(r+1, 0) > 0:
        return (r + 1) * Nc[r+1] / (Nc[r] * r)
    # fallback: no discount (or can use small epsilon)
    return 1.0

def get_Nc(counts_n):
    # counts_n are mapping gram -> count; we want Nc: r -> number of grams with freq r
    return Counter(counts_n.values())

def pkatz(w, h, counts, Nc_dict, k):
    """
    Top (4-gram) Katz probability P_katz(w | h) where h is a tuple of 3 words.
    If c > k -> discounted MLE
    Else -> back off with alpha(h) * lower-order probability
    """
    quad = h + (w,)
    c = counts[4].get(quad, 0)
    Nc = Nc_dict[4]
    if c > k:
        d = discount(c, Nc)
        return d * mle_prob(quad, counts[4], counts[3])
    else:
        alpha = compute_alpha(h, counts, Nc_dict, k, order=4)
        # lower order history for tri is last 2 words of h
        return alpha * pkatz_lower(w, h[1:], counts, Nc_dict, k)

def pkatz_lower(w, h, counts, Nc_dict, k):
    """
    Tri-gram level: h is tuple of 2 words (Wi-2, Wi-3 in your naming)
    If c > k -> discounted MLE on trigram
    Else -> backoff to bigram
    """
    tri = h + (w,)
    c = counts[3].get(tri, 0)
    Nc = Nc_dict[3]
    if c > k:
        d = discount(c, Nc)
        return d * mle_prob(tri, counts[3], counts[2])
    else:
        alpha = compute_alpha(h, counts, Nc_dict, k, order=3)
        return alpha * pkatz_lower2(w, h[1:], counts, Nc_dict, k)

def pkatz_lower2(w, h, counts, Nc_dict, k):
    """
    Bigram level: h is tuple of 1 word.
    If c > k -> discounted MLE on bigram
    Else -> backoff to unigram MLE
    """
    bi = h + (w,)
    c = counts[2].get(bi, 0)
    Nc = Nc_dict[2]
    if c > k:
        d = discount(c, Nc)
        return d * mle_prob(bi, counts[2], counts[1])
    else:
        alpha = compute_alpha(h, counts, Nc_dict, k, order=2)
        # unigram fallback
        return alpha * mle_prob((w,), counts[1], {})

def compute_alpha(h, counts, Nc_dict, k, order=4):
    """
    Compute alpha for history h of length order-1.
    Formula implemented:
      alpha(h) = (1 - sum_{c(w,h) > k} d_{w,h} * P_mle(w|h)) /
                 (1 - sum_{c(w,h) > 0} P_lower(w|h'))
    where P_lower is the backed-off lower-order probability for the corresponding w.
    We iterate over all vocabulary words, but only include terms in the sums when the
    higher-order count condition holds (c > k for numerator; c > 0 for denominator).
    """
    # numerator starts at 1, subtract discounted mass for words with c > k
    numer = 1.0
    denom = 1.0

    # iterate vocabulary (counts[1] keys are ('word',))
    for w_tuple in counts[1].keys():
        wtok = w_tuple[0]
        if order == 4:
            quad = h + (wtok,)
            c = counts[4].get(quad, 0)
            if c > k:
                d = discount(c, Nc_dict[4])
                numer -= d * mle_prob(quad, counts[4], counts[3])
        elif order == 3:
            tri = h + (wtok,)
            c = counts[3].get(tri, 0)
            if c > k:
                d = discount(c, Nc_dict[3])
                numer -= d * mle_prob(tri, counts[3], counts[2])
        elif order == 2:
            bi = h + (wtok,)
            c = counts[2].get(bi, 0)
            if c > k:
                d = discount(c, Nc_dict[2])
                numer -= d * mle_prob(bi, counts[2], counts[1])

    # Denominator: subtract lower-order (backed-off) probabilities
    # but only for words where the higher-order count c > 0 (i.e., observed with this history)
    for w_tuple in counts[1].keys():
        wtok = w_tuple[0]
        if order == 4:
            quad = h + (wtok,)
            c = counts[4].get(quad, 0)
            if c > 0:
                # lower order for quad is trigram history h[1:]
                denom -= pkatz_lower(wtok, h[1:], counts, Nc_dict, k)
        elif order == 3:
            tri = h + (wtok,)
            c = counts[3].get(tri, 0)
            if c > 0:
                # lower order for tri is bigram history h[1:]
                denom -= pkatz_lower2(wtok, h[1:], counts, Nc_dict, k)
        elif order == 2:
            bi = h + (wtok,)
            c = counts[2].get(bi, 0)
            if c > 0:
                # lower order for bigram is unigram mle
                denom -= mle_prob((wtok,), counts[1], {})

    # avoid division by zero
    if denom <= 0:
        return 1.0
    # alpha is the ratio of remaining mass in numerator to remaining mass in denominator
    alpha = numer / denom
    # ensure alpha is non-negative
    if alpha < 0:
        return 1.0
    return alpha

if __name__ == "__main__":
    train_path = Path(TRAIN_FILENAME)
    tokens = list(stream_tokens(train_path))
    counts = count_ngrams(tokens, max_n=4)
    Nc_dict = {n: get_Nc(counts[n]) for n in range(1, 5)}

    # Pick a sample quadgram from training
    quadgrams = list(counts[4].keys())
    if quadgrams:
        h = quadgrams[0][:3]
        w = quadgrams[0][3]
        print("Performing Katz Backoff Probability Calculation for", h, w)
        pk = pkatz(w, h, counts, Nc_dict, k)
        print(f"P_Katz({w} | {h}) = {pk:.12f}")
    else:
        print("No quadgrams found in the data.")
