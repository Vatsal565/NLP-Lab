import re
import math
from collections import Counter, defaultdict

def preprocess(sentence):
    sentence = sentence.lower()

    # URL pattern
    URL_PATTERN = r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&//=]*)"

    tokens = []
    pos = 0

    # Identify URLs and replace them with single token
    for match in re.finditer(URL_PATTERN, sentence):
        start, end = match.span()
        # Text before URL must be tokenized
        before = sentence[pos:start]
        tokens.extend(tokenize_non_url(before))
        tokens.append("URL")
        pos = end
    
    # Remaining part after last URL
    tokens.extend(tokenize_non_url(sentence[pos:]))

    return tokens


def tokenize_non_url(text):
    """
    Tokenize non-URL text and convert:
    - numbers → NUMBER
    - punctuation → PUNCT
    """
    raw_tokens = re.findall(r'\d+|\w+|[^\w\s]', text)

    final_tokens = []
    for tok in raw_tokens:
        if tok.isdigit():
            final_tokens.append("NUMBER")
        elif re.match(r"[^\w\s]", tok):   # punctuation
            final_tokens.append("PUNCT")
        else:
            final_tokens.append(tok)
    return final_tokens


def compute_tf_with_normalization(sentence_tokens, vocab, smoothing=False):
    token_counts = Counter(sentence_tokens)
    total_tokens = len(sentence_tokens)

    tf = {}
    for word in vocab:
        count = token_counts[word]

        if smoothing:
            tf[word] = math.log(1 + (count + 1) / (total_tokens + len(vocab)))
        else:
            if total_tokens > 0:
                tf[word] = math.log(1 + count / total_tokens)
            else:
                tf[word] = 0.0
    return tf


def compute_idf(sentences_tokens, vocab, smoothing=False):
    N = len(sentences_tokens)
    df = defaultdict(int)

    # Count documents containing each term
    for sent in sentences_tokens:
        unique = set(sent)
        for w in unique:
            if w in vocab:
                df[w] += 1

    idf = {}
    for word in vocab:
        if smoothing:
            idf[word] = math.log((N + 1) / (df[word] + 1))
        else:
            if df[word] > 0:
                idf[word] = math.log(N / df[word])
            else:
                idf[word] = 0.0
    return idf


def compute_tf_idf_scores(sentences_tokens):
    # Build vocabulary
    vocab = sorted(set(token for sent in sentences_tokens for token in sent))

    # Compute IDF
    idf = compute_idf(sentences_tokens, vocab, smoothing=True)

    tfidf_scores = []

    for sent in sentences_tokens:
        tf = compute_tf_with_normalization(sent, vocab, smoothing=True)

        tfidf = {word: tf[word] * idf[word] for word in vocab}
        tfidf_scores.append(tfidf)

    return vocab, tfidf_scores


def main():
    sentences = [
        "I bought 2 items from https://example.com/shop!",
        "There are 15 apples, 20 bananas, and 3 mangoes.",
        "Visit www.example.org now.",
    ]

    # PREPROCESSING
    sentences_tokens = [preprocess(s) for s in sentences]

    print("=== Preprocessed Sentences ===")
    for s in sentences_tokens:
        print(s)

    # TF-IDF
    vocab, scores = compute_tf_idf_scores(sentences_tokens)

    print("\n=== Vocabulary ===")
    print(vocab)

    print("\n=== TF-IDF Scores ===")
    for i, score in enumerate(scores):
        print(f"Sentence {i+1}:")
        print(score)


if __name__ == "__main__":
    main()
