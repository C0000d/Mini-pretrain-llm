from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    AutoTokenizer,
)
import math
import torch
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from model_training import (
    EVAL_SIZE,
    CustomDataset,
    CustomArguments,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CKPT_PATH = "./output/checkpoint-30"
BASE_MODEL_PATH = "./models/TinySolar-308m-4k-init"
ORIGINAL_MODEL_NAME = "./models/TinySolar-248m-4k"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def collate_fn(batch):
    ids = [item["input_ids"] for item in batch]
    input_ids = pad_sequence(ids,
                             batch_first=True,
                             padding_value=tokenizer.pad_token_id
                             )

    return {"input_ids": input_ids, "labels": input_ids.clone()}

def perplexity(model, dataloader):
    """measures the uncertainty of a model's predictions. the lower the higher certainty."""
    model.eval()
    nll, n_tok = 0.0, 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss  # the loss straight gives the mean negative-log-likelihood per token, aligns well with the definition of perplexity
            tokens = (batch["input_ids"] != tokenizer.pad_token_id).sum().item()
            nll += loss.item() * tokens
            n_tok += tokens
    return math.exp(nll / n_tok)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
# 1. Load dataset for eval
parser = HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(
    args=["--output_dir", "output"]
)
dataset = CustomDataset(args=args)
eval_dataset = Subset(dataset, range(EVAL_SIZE))

# 2. Load models & tokenizer
model_before = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

tokenizer = AutoTokenizer.from_pretrained(ORIGINAL_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
model_after = AutoModelForCausalLM.from_pretrained(
    CKPT_PATH,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)

eval_loader = DataLoader(
    eval_dataset,
    collate_fn=collate_fn,
    batch_size=2,
    shuffle=False
)

# 3. Compute the perplexity
perplexity_before = perplexity(model_before, eval_loader)
perplexity_after = perplexity(model_after, eval_loader)
print(f"Perplexity before: {round(perplexity_before, 2)}")
print(f"Perplexity after: {round(perplexity_after, 2)}")
print(f"Perplexity improvement: {round(100 * (perplexity_before - perplexity_after) / perplexity_before, 2)}%")