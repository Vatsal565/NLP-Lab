import pickle
from collections import Counter
from math import log10

class NGramLanguageModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counts = Counter()
        self.context_counts = Counter()
        self.vocabulary = set()
        self.vocab_size = 0

    def add_sentence_markers(self, sentence):
        return ['<s>'] * (self.n - 1) + sentence + ['</s>']

    def get_ngrams(self, sentence):
        marked_sentence = self.add_sentence_markers(sentence)
        ngrams = []
        for i in range(len(marked_sentence) - self.n + 1):
            ngram = tuple(marked_sentence[i:i + self.n])
            ngrams.append(ngram)
        return ngrams

    def train(self, sentences):
        for sentence in sentences:
            self.vocabulary.update(sentence)
        self.vocabulary.add('<s>')
        self.vocabulary.add('</s>')
        self.vocab_size = len(self.vocabulary)
        for sentence in sentences:
            ngrams = self.get_ngrams(sentence)
            for ngram in ngrams:
                self.ngram_counts[ngram] += 1
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

    def probability(self, ngram, smoothing='none', k=1):
        if isinstance(ngram, list):
            ngram = tuple(ngram)
        if self.n == 1:
            count = self.ngram_counts[ngram]
            total = sum(self.ngram_counts.values())
            if smoothing == 'add_one':
                return (count + 1) / (total + self.vocab_size)
            elif smoothing == 'add_k':
                return (count + k) / (total + k * self.vocab_size)
            elif smoothing == 'add_token_type':
                unique_types = len(self.ngram_counts)
                return (count + 1) / (total + unique_types)
            else:
                return count / total if total > 0 else 0
        else:
            context = ngram[:-1]
            count = self.ngram_counts[ngram]
            context_count = self.context_counts[context]
            if smoothing == 'add_one':
                return (count + 1) / (context_count + self.vocab_size)
            elif smoothing == 'add_k':
                return (count + k) / (context_count + k * self.vocab_size)
            elif smoothing == 'add_token_type':
                unique_types = len([ng for ng in self.ngram_counts.keys() if ng[:-1] == context])
                if unique_types == 0:
                    unique_types = 1
                return (count + 1) / (context_count + unique_types)
            else:
                return count / context_count if context_count > 0 else 0
