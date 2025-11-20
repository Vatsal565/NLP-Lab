import re
import json
from collections import Counter

MERGE_STEPS = 1000  
VOCAB_SIZE = 32000

def tokenize(text):
    return re.findall(r'\w+|[^\w\s]', text.lower())

def get_stats(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    bigram = re.escape(' '.join(pair))
    replacement = ''.join(pair)
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    
    new_vocab = {}
    for word, freq in vocab.items():
        new_word = p.sub(replacement, word)
        new_vocab[new_word] = freq
    return new_vocab

def train_bpe(text_corpus):
    print("Initializing vocabulary...")
    tokens = tokenize(text_corpus)
    vocab = Counter([" ".join(list(t)) + " </w>" for t in tokens])
    
    merges = []
    
    for i in range(MERGE_STEPS):
        pairs = get_stats(vocab)
        if not pairs:
            break

        best_pair = pairs.most_common(1)[0][0]
        merges.append(best_pair)
        vocab = merge_vocab(best_pair, vocab)
        
        if (i + 1) % 100 == 0:
            print(f"Step {i+1}/{MERGE_STEPS} | Merged: {best_pair}")

        unique_tokens = set(' '.join(vocab.keys()).split())
        if len(unique_tokens) >= VOCAB_SIZE:
            break
            
    return merges, vocab

def bpe_encode(text, merges):
    encoded_tokens = []
    words = tokenize(text.lower())
    
    for word in words:
        word_str = " ".join(list(word)) + " </w>"
        
        for pair in merges:
            bigram = re.escape(' '.join(pair))
            p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
            word_str = p.sub(''.join(pair), word_str)
            
        encoded_tokens.extend(word_str.split())
        
    return encoded_tokens

corpus_text = """
The quick brown fox jumps over the lazy dog. 
The dog was not amused by the fox.
Deep learning is a subset of machine learning.
"""

merges, final_vocab = train_bpe(corpus_text)

model = {'merges': merges, 'vocab': list(final_vocab.keys())}
with open('bpe_en_model.json', 'w') as f:
    json.dump(model, f)

print("\n--- Testing ---")
test_str = "The machine is learning fast."
encoded = bpe_encode(test_str, merges)
print(f"Input: {test_str}")
print(f"Tokens: {encoded}")