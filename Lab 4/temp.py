import pandas as pd

df = pd.read_parquet("gujarati_tokenized_batch_0000.parquet")
df.to_csv('out.txt', index=False, sep=' ') 
