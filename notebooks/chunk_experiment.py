#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tiktoken


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks based on tiktoken tokens.

    Args:
        text: Input text to chunk
        chunk_size: Target tokens per chunk
        overlap: Overlapping tokens between chunks (helps with boundary effects)

    Returns:
        List of text chunks

    """
    enc = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding

    # Tokenize full text
    tokens = enc.encode(text)

    # If shorter than chunk_size, return as-is
    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    stride = chunk_size - overlap  # Move forward by this much each time

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]

        # Decode back to text
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move to next chunk
        if end >= len(tokens):
            break
        start += stride

    return chunks


# Test it
sample = "This is a test. " * 1000  # Long text
chunks = chunk_text(sample, chunk_size=300, overlap=50)
print(f"Split into {len(chunks)} chunks")
print(f"First chunk length: {len(tiktoken.get_encoding('cl100k_base').encode(chunks[0]))} tokens")
print(f"Last chunk length: {len(tiktoken.get_encoding('cl100k_base').encode(chunks[-1]))} tokens")


# In[ ]:


chunks[0]


# In[ ]:


import polars as pl
from tqdm import tqdm


def create_chunked_dataset(df: pl.DataFrame, chunk_size: int = 500, overlap: int = 50) -> pl.DataFrame:
    """Version with document ID tracking."""
    rows = []

    for doc_id, row in enumerate(tqdm(df.iter_rows(named=True), total=len(df))):
        text = row["text"]
        label = row["label"]
        dataset = row.get("dataset", "unknown")

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk_idx, chunk in enumerate(chunks):
            rows.append({
                "doc_id": doc_id,
                "text": chunk,
                "label": label,
                "dataset": dataset,
                "chunk_idx": chunk_idx,
                "num_chunks": len(chunks),
            })

    return pl.DataFrame(rows)


# Apply to your data
df_train = pl.read_parquet("../data/curated_dataset_train.parquet")
df_test = pl.read_parquet("../data/curated_dataset_test.parquet")

print(f"Original train samples: {len(df_train)}")
df_train_chunked = create_chunked_dataset(df_train, chunk_size=300, overlap=50)
print(f"Chunked train samples: {len(df_train_chunked)}")

print(f"Original test samples: {len(df_test)}")
df_test_chunked = create_chunked_dataset(df_test, chunk_size=300, overlap=50)
print(f"Chunked test samples: {len(df_test_chunked)}")

# Check expansion ratio
print(f"\nExpansion ratio: {len(df_train_chunked) / len(df_train):.2f}x")

