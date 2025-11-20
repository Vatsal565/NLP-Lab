import heapq
from collections import defaultdict, Counter

END = "</w>"

class OptimizedBPE:
    def __init__(self):
        self.merges = []

    # ------------------------------------------
    # Tokenize text into words and convert each
    # into a list of characters + end marker.
    # ------------------------------------------
    def build_vocab(self, text):
        words = text.lower().split()
        vocab = Counter()

        for w in words:
            tokens = list(w) + [END]
            vocab[tuple(tokens)] += 1

        return vocab

    # ------------------------------------------
    # Count initial token pair frequencies
    # ------------------------------------------
    def get_pair_stats(self, vocab):
        pair_freq = defaultdict(int)

        for tokens, freq in vocab.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_freq[pair] += freq

        return pair_freq

    # ------------------------------------------
    # Replace occurrences of a pair in a token list
    # ------------------------------------------
    def merge_pair_in_word(self, word, pair):
        new_word = []
        i = 0
        a, b = pair
        ab = a + b

        while i < len(word):
            if i < len(word) - 1 and word[i] == a and word[i+1] == b:
                new_word.append(ab)
                i += 2
            else:
                new_word.append(word[i])
                i += 1

        return tuple(new_word)

    # ------------------------------------------
    # Train BPE with optimized incremental updates
    # ------------------------------------------
    def train(self, text, num_merges=1000):
        vocab = self.build_vocab(text)
        pair_freq = self.get_pair_stats(vocab)

        # Max-heap for best pair selection
        # Python heapq is min-heap → store negative counts
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        for step in range(num_merges):

            # Pop until we get a valid (non-stale) pair
            while heap:
                neg_freq, best_pair = heapq.heappop(heap)
                if pair_freq.get(best_pair, 0) == -neg_freq:
                    break
            else:
                print("No more pairs to merge.")
                break

            self.merges.append(best_pair)
            a, b = best_pair
            ab = a + b

            # Update vocab: merge only affected words
            new_vocab = Counter()
            changed_words = []

            for word, freq in vocab.items():
                if best_pair in zip(word, word[1:]):
                    new_word = self.merge_pair_in_word(word, best_pair)
                    new_vocab[new_word] += freq
                    changed_words.append((word, new_word, freq))
                else:
                    new_vocab[word] += freq

            vocab = new_vocab

            # Update pair frequencies incrementally
            pair_freq = defaultdict(int)

            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair_freq[(word[i], word[i+1])] += freq

            # Rebuild heap from updated stats
            heap = [(-freq, pair) for pair, freq in pair_freq.items()]
            heapq.heapify(heap)

            if (step + 1) % 50 == 0:
                print(f"Merge {step+1}: {best_pair}")

        return self.merges

    # ------------------------------------------
    # Apply learned merges to encode new text
    # ------------------------------------------
    def encode(self, word):
        tokens = list(word.lower()) + [END]

        for a, b in self.merges:
            i = 0
            new_tokens = []
            ab = a + b

            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == a and tokens[i+1] == b:
                    new_tokens.append(ab)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = new_tokens

        return tokens


# ------------------ Example Usage ------------------

if __name__ == "__main__":
    corpus = """
    The quick brown fox jumps over the lazy dog.
    The dog was not amused by the fox.
    Deep learning is a subset of machine learning.
    """

    bpe = OptimizedBPE()
    merges = bpe.train(corpus, num_merges=200)

    print("\nLearned merges:", merges[:30])

    test = "The machine is learning fast."
    print("\nEncoding:", test)
    print("Tokens:", bpe.encode(test))
