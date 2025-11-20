import os
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

LAB7_PATH = Path(__file__).parent
TRAIN_PATH = LAB7_PATH / "train.txt"
VAL_PATH = LAB7_PATH / "val.txt"
TEST_PATH = LAB7_PATH / "test.txt"
TFIDF_TRAIN_NPZ = LAB7_PATH / "tfidf_train.npz"
TFIDF_VAL_NPZ = LAB7_PATH / "tfidf_val.npz"
TFIDF_TEST_NPZ = LAB7_PATH / "tfidf_test.npz"
VAL_NN_PATH = LAB7_PATH / "val_neighbors.txt"
TEST_NN_PATH = LAB7_PATH / "test_neighbors.txt"

def load_sentences(path):
    with open(path, encoding='utf-8') as f:
        return [line.rstrip("\n") for line in f]

def filter_csr_columns(csr, valid_columns):
    coo = csr.tocoo()
    col_numpy = coo.col  
    mask = np.isin(col_numpy, list(valid_columns))  # Use numpy's isin for efficient membership check
    filtered_data = coo.data[mask]
    filtered_row = coo.row[mask]
    filtered_col = coo.col[mask]
    new_col_indices = {col: idx for idx, col in enumerate(sorted(valid_columns))}
    filtered_col = np.asarray([new_col_indices[int(col)] for col in filtered_col])  # Map to new indices
    shape = (csr.shape[0], len(valid_columns))
    return sp.csr_matrix((filtered_data, (filtered_row, filtered_col)), shape=shape)

def load_csr_npz(path, valid_columns=None):
    npz = np.load(path, allow_pickle=False)
    data = npz["data"]
    indices = npz["indices"].astype(np.int32)
    indptr = npz["indptr"].astype(np.int32)
    shape = tuple(npz["shape"].astype(int).tolist())
    csr = sp.csr_matrix((data, indices, indptr), shape=shape)

    if valid_columns is not None:
        csr = filter_csr_columns(csr, valid_columns)
    return csr

def knn_cpu(train_csr, train_norms, queries_csr, query_norms, train_sentences, query_sentences, out_path, topk=5, batch_size=128):
    n_queries = queries_csr.shape[0]
    with open(out_path, 'w', encoding='utf-8') as fout:
        for start in range(0, n_queries, batch_size):
            end = min(n_queries, start + batch_size)
            batch = queries_csr[start:end]
            batch_norm = query_norms[start:end]
            sims = batch.dot(train_csr.T).toarray()  # (batch, n_train) dense numpy array
            denom = np.outer(batch_norm, train_norms)
            sims = sims / (denom + 1e-12)
            k = min(topk, sims.shape[1])
            idx_part = np.argpartition(-sims, k-1, axis=1)[:, :k]
            for i in range(sims.shape[0]):
                row_idx = int(start + i)
                candidates = idx_part[i]
                scores = sims[i, candidates]
                order = scores.argsort()[::-1]
                sorted_idx = candidates[order]
                sorted_scores = scores[order]
                fout.write(f"Query [{row_idx}]: {query_sentences[row_idx]}\n")
                for rank, (tidx, sc) in enumerate(zip(sorted_idx, sorted_scores), start=1):
                    fout.write(f"\tNeighbor {rank}: TrainIndex={int(tidx)} Score={float(sc):.6f} Sentence={train_sentences[int(tidx)]}\n")
                fout.write("\n")

def row_norms_csr(csr):
    sq = csr.multiply(csr)
    s = np.asarray(sq.sum(axis=1)).ravel()
    return np.sqrt(s + 1e-12)

train_sentences = load_sentences(TRAIN_PATH)
val_sentences = load_sentences(VAL_PATH)
test_sentences = load_sentences(TEST_PATH)

print("Loading TF-IDF matrices...")
train_csr = load_csr_npz(TFIDF_TRAIN_NPZ)
val_csr = load_csr_npz(TFIDF_VAL_NPZ)
test_csr = load_csr_npz(TFIDF_TEST_NPZ)

valid_columns = set(val_csr.indices).union(set(test_csr.indices))

train_csr = filter_csr_columns(train_csr, valid_columns)
val_csr = filter_csr_columns(val_csr, valid_columns)
test_csr = filter_csr_columns(test_csr, valid_columns)

print("Filtered Shapes:", train_csr.shape, val_csr.shape, test_csr.shape)

train_norms = row_norms_csr(train_csr)
val_norms = row_norms_csr(val_csr)
test_norms = row_norms_csr(test_csr)

print("Computing val")
knn_cpu(train_csr, train_norms, val_csr, val_norms, train_sentences, val_sentences, VAL_NN_PATH, topk=5, batch_size=128)
print(f"Saved {VAL_NN_PATH}")

print("Computing test")
knn_cpu(train_csr, train_norms, test_csr, test_norms, train_sentences, test_sentences, TEST_NN_PATH, topk=5, batch_size=128)
print(f"Saved {TEST_NN_PATH}")