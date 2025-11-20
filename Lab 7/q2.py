import random
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import json

LAB7_PATH = Path(__file__).parent
TOKENIZED_PATH = LAB7_PATH / "indiccorp_gu_tokenized.txt"
TRAIN_PATH = LAB7_PATH / "train.txt"
VAL_PATH = LAB7_PATH / "val.txt"
TEST_PATH = LAB7_PATH / "test.txt"
TFIDF_OUTPUT_PATH = LAB7_PATH / "tfidf_examples.txt"
TFIDF_TRAIN_NPZ = LAB7_PATH / "tfidf_train.npz"
TFIDF_VAL_NPZ = LAB7_PATH / "tfidf_val.npz"
TFIDF_TEST_NPZ = LAB7_PATH / "tfidf_test.npz"
VOCAB_PATH = LAB7_PATH / "vocab.json"
IDF_PATH = LAB7_PATH / "idf.npy"

def sentence_generator(filepath):
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def split_sentences(sentences, train_ratio=0.8, val_ratio=0.1, seed=42):
    sentences = list(sentences)
    random.seed(seed)
    random.shuffle(sentences)
    n = len(sentences)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = sentences[:n_train]
    val = sentences[n_train:n_train+n_val]
    test = sentences[n_train+n_val:]
    return train, val, test

def build_tf_matrix(sentences, vocab=None, batch_size=100, default_idf=1.0):
    if vocab is None:
        vocab = {}
        next_idx = 0
    else:
        next_idx = max(vocab.values()) + 1 if vocab else 0

    all_rows = []
    all_cols = []
    all_data = []
    docs_tokens = []

    for batch_start in range(0, len(sentences), batch_size):
        batch_sentences = sentences[batch_start:batch_start + batch_size]
        rows = []
        cols = []
        data = []

        for doc_id, s in enumerate(batch_sentences, start=batch_start):
            tokens = s.split()
            docs_tokens.append(tokens)
            if not tokens:
                continue
            counts = {}
            for t in tokens:
                if t not in vocab:
                    # Add new token to the vocabulary with smoothing
                    vocab[t] = next_idx
                    next_idx += 1
                idx = vocab[t]
                counts[idx] = counts.get(idx, 0) + 1
            doc_len = len(tokens)
            for idx, cnt in counts.items():
                rows.append(doc_id)
                cols.append(idx)
                data.append(float(cnt) / doc_len)  # TF = count / doc length

        # Append batch data to global lists
        all_rows.extend(rows)
        all_cols.extend(cols)
        all_data.extend(data)

    # Construct the sparse matrix on the CPU
    n_docs = len(sentences)
    n_vocab = max(vocab.values()) + 1
    rows_arr = np.asarray(all_rows, dtype=np.int32)
    cols_arr = np.asarray(all_cols, dtype=np.int32)
    data_arr = np.asarray(all_data, dtype=np.float32)
    tf_coo = sp.coo_matrix((data_arr, (rows_arr, cols_arr)), shape=(n_docs, n_vocab), dtype=np.float32)
    return tf_coo, vocab, docs_tokens

def compute_idf_from_tf_coo(tf_coo, n_docs):
    csr = tf_coo.tocsr()
    df_col = np.asarray((csr > 0).sum(axis=0)).ravel().astype(np.float32)
    idf = np.log((n_docs + 1.0) / (df_col + 1.0)) + 1.0
    return idf

def apply_idf_to_coo(tf_coo, idf, default_idf=1.0):
    coo = tf_coo.tocoo()
    extended_idf = np.full(coo.shape[1], default_idf, dtype=np.float32)
    extended_idf[:len(idf)] = idf  # Extend idf with default values for new tokens
    coo.data = coo.data * extended_idf[coo.col]
    return coo.tocsr()

def row_norms_csr(csr):
    # returns 1D numpy array of L2 norms per row
    sq = csr.multiply(csr)
    s = np.asarray(sq.sum(axis=1)).ravel()
    return np.sqrt(s + 1e-12)

def save_tfidf_examples_from_matrix(vocab, X_train_csr, train_sentences, output_path, num_examples=5):
    inv_vocab = [None] * (max(vocab.values()) + 1)
    for t, idx in vocab.items():
        inv_vocab[idx] = t
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("TF-IDF feature names (first 50):\n")
        f.write(', '.join(inv_vocab[:50]) + ' ...\n\n')
        f.write("TF-IDF vectors (first {} examples):\n".format(num_examples))
        for i in range(min(num_examples, X_train_csr.shape[0])):
            f.write(f"Sentence: {train_sentences[i]}\n")
            row = X_train_csr.getrow(i)
            if row.nnz == 0:
                f.write("TF-IDF: []\n\n")
                continue
            cols = row.indices
            vals = row.data
            nonzero = [(inv_vocab[int(cols[j])], float(vals[j])) for j in range(len(cols))]
            f.write("TF-IDF: " + str(nonzero) + "\n\n")

def save_csr_npz(path, csr):
    csr = csr.tocsr()
    np.savez_compressed(
        path,
        data=csr.data,
        indices=csr.indices,
        indptr=csr.indptr,
        shape=np.array(csr.shape, dtype=np.int64),
    )

def save_vocab_json(path, vocab):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False)

# Step 1: Skip tokenization and splitting, use existing train/val/test files
print(f"Using existing train, val, and test files: {TRAIN_PATH}, {VAL_PATH}, {TEST_PATH}")

# Load sentences from existing files
train = list(sentence_generator(TRAIN_PATH))[:10_00_000]  # Limit to 10 lakh sentences
val = list(sentence_generator(VAL_PATH))[:1_000]          # Limit to 1,000 sentences
test = list(sentence_generator(TEST_PATH))[:1_000]        # Limit to 1,000 sentences
print(f"Loaded Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Step 2: Manual CPU TF-IDF (train vocab -> apply idf to val/test)
train_tf_coo, vocab, train_token_lists = build_tf_matrix(train, vocab=None)
n_train = train_tf_coo.shape[0]
idf = compute_idf_from_tf_coo(train_tf_coo, n_train)
train_tfidf = apply_idf_to_coo(train_tf_coo, idf)  # CSR

# Use the same vocabulary for validation and test sets
val_tf_coo, _, _ = build_tf_matrix(val, vocab=vocab)
test_tf_coo, _, _ = build_tf_matrix(test, vocab=vocab)
val_tfidf = apply_idf_to_coo(val_tf_coo, idf)
test_tfidf = apply_idf_to_coo(test_tf_coo, idf)

print("TF-IDF shapes (sparse CSR):")
print("Train:", train_tfidf.shape)
print("Val:", val_tfidf.shape)
print("Test:", test_tfidf.shape)

# Step 3: Save examples (first few)
save_tfidf_examples_from_matrix(vocab, train_tfidf, train, TFIDF_OUTPUT_PATH)
print(f"TF-IDF examples saved to {TFIDF_OUTPUT_PATH}")

# Step 4: Persist TF-IDF matrices + vocab + idf for Q3 nearest-neighbour step
print("Saving TF-IDF matrices and metadata to disk for Q3...")
save_csr_npz(TFIDF_TRAIN_NPZ, train_tfidf)
save_csr_npz(TFIDF_VAL_NPZ, val_tfidf)
save_csr_npz(TFIDF_TEST_NPZ, test_tfidf)
save_vocab_json(VOCAB_PATH, vocab)
np.save(IDF_PATH, idf)
print(f"Saved: {TFIDF_TRAIN_NPZ}, {TFIDF_VAL_NPZ}, {TFIDF_TEST_NPZ}, {VOCAB_PATH}, {IDF_PATH}")