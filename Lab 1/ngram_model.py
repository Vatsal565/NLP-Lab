from collections import Counter
from tokenizer import tokenize_sentence, tokenizer
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
    

with open("msd.txt") as f:
    text = f.read()
    sentences = tokenize_sentence(text)
    sentences = [tokenizer(sent) for sent in sentences]
    trigram_model = NGramModel(3)
    trigram_model.train(sentences)


with open("msd.txt") as f:
    text = f.read()
    sentences = tokenize_sentence(text)
    sentences = [tokenizer(sent) for sent in sentences]
    unigram_model = NGramModel(1)
    unigram_model.train(sentences)


import math
trigrams = list(trigram_model.ngram_counter.keys())

pmi_scores = []
for w1,w2,w3 in trigrams:
    c_w1w2w3 = trigram_model.ngram_counter.get((w1, w2, w3), 0)
    c_w1 = unigram_model.ngram_counter.get((w1, ), 0)
    c_w2 = unigram_model.ngram_counter.get((w2, ), 0)
    c_w3 = unigram_model.ngram_counter.get((w3, ), 0)

    if c_w1 == 0 or c_w2 == 0 or c_w3 == 0:
        pmi = None
    
    else:
        p_w1 = c_w1 / sum(unigram_model.ngram_counter.values())
        p_w2 = c_w2 / sum(unigram_model.ngram_counter.values())
        p_w3 = c_w3 / sum(unigram_model.ngram_counter.values())
        p_w1w2w3 = c_w1w2w3 / sum(trigram_model.ngram_counter.values())
        denom = p_w1 * p_w2 * p_w3
        if denom == 0 or p_w1w2w3 == 0:
            pmi = None
        else:
            pmi = math.log(p_w1w2w3 / denom)
        
    pmi_scores.append((w1, w2, w3, c_w1w2w3, c_w1, c_w2, c_w3, pmi))

print(pmi_scores)


pmi_scores2 = [row for row in pmi_scores if row[-1] is not None]
pmi_scores2.sort(key=lambda x: x[-1], reverse=True)
pmi_scores2[-10:]





# sentence prob
models = {
    'Unigram': NGramModel(1),
    'Bigram': NGramModel(2),
    'Trigram': NGramModel(3),
    'Quadrigram': NGramModel(4)
}

with open("msd.txt") as f:
    text = f.read()
    sentences = tokenize_sentence(text)
    sentences = [tokenizer(sent) for sent in sentences]

for name, model in models.items():
    model.train(sentences)
    print(f"Trained {name} model (vocab size: {model.vocab_size})")

smoothings = ['none', 'add_one', 'add_k']
total_sentences = len(sentences)
results = []
for name, model in models.items():
    for sm in smoothings:
        k = 1 if sm == 'add_k' else None  # Only for add_k
        total_logp = 0.0
        impossible_count = 0
        for sent in sentences:
            logp = model.sentence_log_probability(sent, smoothing=sm, k=k) # type: ignore
            if math.isinf(logp) and logp < 0:
                impossible_count += 1
            total_logp += logp
        avg_logp = total_logp / total_sentences if total_sentences > 0 else 0
        perplexity = math.exp(-avg_logp) if avg_logp > float('-inf') else float('inf')
        results.append({
            'Model': name,
            'Smoothing': sm,
            'Avg Log Prob': round(avg_logp, 4),
            'Perplexity': round(perplexity, 2) if not math.isinf(perplexity) else 'inf',
            'Impossible Sentences': impossible_count
        })

see = 'Dhoni finished the season with 283 runs in 5 matches.'
model.sentence_log_probability(tokenizer(see))

print("\nEvaluation Results (on 1000 test sentences):")
print("| Model       | Smoothing   | Avg Log Prob | Perplexity | Impossible Sents |")
print("|-------------|-------------|--------------|------------|------------------|")
for res in results:
    print(f"| {res['Model']:<11} | {res['Smoothing']:<11} | {res['Avg Log Prob']:<12} | {res['Perplexity']:<10} | {res['Impossible Sentences']:<16} |")