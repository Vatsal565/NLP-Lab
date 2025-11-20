import random
import pandas as pd

# Path to parquet file
input_path = "sentence_tokenizer.parquet"

# Read parquet file
df = pd.read_parquet(input_path)

# Assuming sentences are in a column named "sentence"
sentences = df["tokenized_sentence"].astype(str).tolist()

# Shuffle with fixed seed for reproducibility
random.seed(42)
random.shuffle(sentences)

# Split sizes
val_size = 1000
test_size = 1000

val_sentences = sentences[:val_size]
test_sentences = sentences[val_size:val_size + test_size]
train_sentences = sentences[val_size + test_size:]

# Save splits
with open("train.txt", "w", encoding="utf-8") as f:
    f.writelines(s + "\n" for s in train_sentences)

with open("val.txt", "w", encoding="utf-8") as f:
    f.writelines(s + "\n" for s in val_sentences)

with open("test.txt", "w", encoding="utf-8") as f:
    f.writelines(s + "\n" for s in test_sentences)

print(f"Train: {len(train_sentences)}, Val: {len(val_sentences)}, Test: {len(test_sentences)}")
