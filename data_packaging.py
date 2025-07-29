import datasets
from transformers import AutoTokenizer
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tokenization(example):
    # Tokenize ONE row
    # split raw text into sub-word tokens
    tokens = tokenizer.tokenize(example['text'])

    # convert those tokens to integer IDs
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Wrap with <bos> and <eos> special tokens
    tokens_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]

    # Store results back into the example dict
    example["input_ids"] = tokens_ids
    example["num_tokens"] = len(tokens_ids)
    return example


# ---------------------------------------------------------------------------
# Data Packaging
# ---------------------------------------------------------------------------
# Load the preprocessed dataset
dataset = datasets.load_dataset(
    "parquet",
    data_files="./data/preprocessed_dataset.parquet",
    split="train"
    )
print("load dataset", dataset)

# split the dataset into 10 equal-sized shards and return just one of them
dataset = dataset.shard(num_shards=10, index=0)  # keep only shard #0
print("sharded dataset", dataset)

# load tokenizer
model_path_or_name = 'upstage/SOLAR-10.7B-v1.0'  # upstage AI, 10.7 billion params, decoder-only stack
tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name,
    use_fast=False
)

dataset = dataset.map(tokenization, load_from_cache_file=False) # map calls tokenizer for every row, collect the returned dicts into a brand-new Dataset where new columns.
# print(dataset)
#
# # We can inspect a sample from the dataset
# sample = dataset[3]
# print("text", sample["text"][:30])
# print("input_ids", sample["input_ids"][:30])
# print("num_tokens", sample["num_tokens"])
# print("total num of tokens", np.sum(dataset["num_tokens"]))

# Packing the data, GPU friendly
input_ids = np.concatenate(dataset["input_ids"])   # flatten list-of-lists to 1-D array
print("num of tokens before fix length", len(input_ids))

# decoder-only model with a casual-LM loss, you usually feed it fixed-length blocks of tokens
# clip the tail so the length is a clean multiple of 32
max_seq_length = 32
total_length = len(input_ids) - len(input_ids) % max_seq_length
print("num of tokens after fix length", total_length)

# drop the left-over tokens that don't fit
input_ids = input_ids[:total_length]
# print(input_ids.shape)

input_ids_reshaped = (
    input_ids
    .reshape(-1, max_seq_length)     # 1-D -> 2-D (num_rows, 32), -1 let numpy infer the first dimension len(input_ids)//32
    .astype(np.int32)                # cast from int64 to int32 to save RAM
)
# print(input_ids_reshaped.shape)
#
# print("type of input_ids_reshaped: ", type(input_ids_reshaped))

# convert to hugging face dataset
input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)
print(packaged_pretrain_dataset)

# save the packed dataset to disk
packaged_pretrain_dataset.to_parquet("./data/packaged_pretrain_dataset.parquet")
