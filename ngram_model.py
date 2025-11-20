from collections import Counter
import math

class NGramModel:
    def __init__(self, n=1):
        self.n = n
        self.ngram_counter = Counter()
        self.context_counter = Counter()
        self.vocab = set()
        self.vocab_size = 0

    def add_sentence_markers(self, sentence):
        return (self.n - 1) * ['<s>'] + sentence + ['</s>']

    def get_ngrams(self, sentence):
        marked_sentence = self.add_sentence_markers(sentence)
        ngrams = []
        for i in range(len(marked_sentence)- self.n + 1):
            ngram = tuple(marked_sentence[i: i+self.n])
            ngrams.append(ngram)
        return ngrams

    def train(self, sentences):
        for sentence in sentences:
            self.vocab.update(sentence)
        
        self.vocab.add('<s>')
        self.vocab.add('</s>')

        self.vocab_size = len(self.vocab)

        for sentence in sentences:
            ngrams = self.get_ngrams(sentence)
            
            for ngram in ngrams:
                self.ngram_counter[ngram] += 1
                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counter[context] += 1
    
    def probability(self, ngram, smoothing='none', k=1):
        if isinstance(ngram, list):
            ngram = tuple(ngram)
        
        if self.n == 1:
            count = self.ngram_counter[ngram]
            total = sum(self.ngram_counter.values())
            if smoothing == 'add_one':
                return (count + 1) / (total + self.vocab_size)
            elif smoothing == 'add_k':
                return (count + k) / (total + k * self.vocab_size)
            elif smoothing == 'add_token_type':
                unique = len(self.ngram_counter)
                return (count + 1) / (total + unique)
            else:
                return count / total if total > 0 else 0
        else:
            context = ngram[:-1]
            count = self.ngram_counter[ngram]
            context_count = self.context_counter[context]
            if smoothing == 'add_one':
                return (count + 1) / (context_count + self.vocab_size)
            elif smoothing == 'add_k':
                return (count + k) / (context_count + k * self.vocab_size)
            elif smoothing == 'add_token_type':
                unique_types = len([ng for ng in self.ngram_counter.keys() if ng[:-1] == context])
                if unique_types == 0:
                    unique_types = 1
                return (count + 1) / (context_count + unique_types)
            else:
                return count / context_count if context_count > 0 else 0
    
    def sentence_log_probability(self, sentence, smoothing='none', k=1):
        """Compute log probability of a sentence as sum log P(ngram) over n-grams."""
        ngrams = self.get_ngrams(sentence)
        log_prob = 0.0
        for ngram in ngrams:
            p = self.probability(ngram, smoothing, k)
            if p > 0:
                log_prob += math.log(p)
            else:
                log_prob += float('-inf')  # Sentence impossible
                break
        return log_prob