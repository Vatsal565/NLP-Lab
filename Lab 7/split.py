import random

input_path = "F:\\NLP Lab\\Lab 7\\indiccorp_gu_tokenized.txt"

# Read all sentences
with open(input_path, "r", encoding="utf-8") as f:
    sentences = f.readlines()

# Shuffle deterministically
random.seed(42)
random.shuffle(sentences)

# Sizes
train_size = 17438656
val_size = 2179832
test_size = 2179832

# Split
val_sentences = sentences[:val_size]
test_sentences = sentences[val_size:val_size+test_size]
train_sentences = sentences[val_size+test_size:val_size+test_size+train_size]

# Write splits
with open("train.txt", "w", encoding="utf-8") as f:
    f.writelines(train_sentences)

with open("val.txt", "w", encoding="utf-8") as f:
    f.writelines(val_sentences)

with open("test.txt", "w", encoding="utf-8") as f:
    f.writelines(test_sentences)

print(f"Train: {len(train_sentences)}, Val: {len(val_sentences)}, Test: {len(test_sentences)}")
