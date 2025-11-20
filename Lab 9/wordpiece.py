import re
from collections import Counter

# Constants
ITR = 1000
VOCAB_SIZE = 32000
SUBWORD_PREFIX = "##"

def tokenize(text):
    """Simple tokenization: words and non-word characters."""
    return re.findall(r"\w+|[^\w\s]", text.lower())

def initialize_word_symbols(word_freqs):
    """Splits words into initial symbols (first char, then '##' + rest)."""
    ws = {}
    for w, f in word_freqs.items():
        if not w: continue
        ws[w] = [w[0]] + [SUBWORD_PREFIX + c for c in w[1:]]
    return ws

def get_pair_counts(word_symbols, word_freqs):
    """Counts adjacent symbol pairs weighted by word frequency."""
    return Counter(p for w, syms in word_symbols.items() 
                   for p in zip(syms, syms[1:]) for _ in range(word_freqs[w]))

def merge_symbols(pair, word_symbols):
    """Merges a pair across all words and returns the new symbol."""
    s1, s2 = pair
    new_sym = s1 + (s2[len(SUBWORD_PREFIX):] if s2.startswith(SUBWORD_PREFIX) else s2)
    
    new_ws = {}
    for w, syms in word_symbols.items():
        i, new_syms = 0, []
        while i < len(syms):
            if i < len(syms) - 1 and syms[i] == s1 and syms[i + 1] == s2:
                new_syms.append(new_sym)
                i += 2
            else:
                new_syms.append(syms[i])
                i += 1
        new_ws[w] = new_syms
    return new_ws, new_sym

def find_best_merge(pairs, word_symbols, word_freqs):
    """Finds the best pair based on the WordPiece score (prob. ratio)."""
    sym_counts = Counter(s for w, syms in word_symbols.items() for s in syms 
                         for _ in range(word_freqs[w]))
    
    T_pairs = sum(pairs.values())
    T_syms = sum(sym_counts.values())
    if not T_pairs or not T_syms: return None, 0

    best_pair, best_score = None, float('-inf')

    for (x, y), freq_xy in pairs.items():
        if freq_xy <= 1: continue
        p_x, p_y = sym_counts[x] / T_syms, sym_counts[y] / T_syms
        
        if p_x > 0 and p_y > 0:
            score = (freq_xy / T_pairs) / (p_x * p_y)
            if score > best_score:
                best_score = score
                best_pair = (x, y)
                
    return best_pair, best_score

def train_wordpiece(corpus_path):
    """The main WordPiece training loop."""
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            raw_text = f.read().lower()
    except FileNotFoundError:
        print(f"Error: Training file not found at {corpus_path}")
        return [], {}, {}
    
    word_freqs = Counter(tokenize(raw_text))
    word_symbols = initialize_word_symbols(word_freqs)
    
    vocab = {s for syms in word_symbols.values() for s in syms}
    
    print(f"Initial vocab size: {len(vocab)}")

    merged_pairs = []
    for i in range(ITR):
        if len(vocab) >= VOCAB_SIZE:
            print(f"Reached target vocab size {len(vocab)} at step {i}")
            break

        pairs = get_pair_counts(word_symbols, word_freqs)
        if not pairs:
            print(f"No more pairs to merge at step {i}")
            break
        
        best_pair, score = find_best_merge(pairs, word_symbols, word_freqs)
        
        if best_pair is None or score <= 1: # Added score check for significance
            print(f"No statistically significant pair found at step {i}")
            break
            
        word_symbols, new_symbol = merge_symbols(best_pair, word_symbols)
        merged_pairs.append((best_pair, new_symbol))
        vocab.add(new_symbol)

        if (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{ITR} | New Token: {new_symbol} | Score: {score:.2f}")

    print(f"\nTraining completed. Total merges: {len(merged_pairs)}")
    return merged_pairs, vocab, word_symbols

def wordpiece_tokenize(sentence, final_vocab):
    """WordPiece encoding using greedy longest-match-first."""
    out = []
    for word in tokenize(sentence):
        if not word: continue
        
        i, word_pieces = 0, []
        while i < len(word):
            match, advance = None, 1 # Fallback to single character
            
            # Find longest matching sub-string
            for j in range(len(word), i, -1):
                candidate = word[i:j]
                token = candidate if i == 0 else SUBWORD_PREFIX + candidate
                
                if token in final_vocab:
                    match = token
                    advance = len(candidate)
                    break # Longest match found
            
            # Use matched token or fallback
            matched_token = match if match else (word[i] if i == 0 else SUBWORD_PREFIX + word[i])
            word_pieces.append(matched_token)
            i += advance
        out.extend(word_pieces)
    return out

MERGED_PAIRS, FINAL_VOCAB, WORD_SYMBOLS = train_wordpiece("train_sampled.txt")

# Test encoding
TEST_TEXT_EN = "The machine is learning fast."
TEST_TOKENS = wordpiece_tokenize(TEST_TEXT_EN, FINAL_VOCAB)

print("\n--- Results ---")
print(f"Final Vocab Size: {len(FINAL_VOCAB)}")
print(f"WordPiece Tokens for '{TEST_TEXT_EN}':")
print(TEST_TOKENS)