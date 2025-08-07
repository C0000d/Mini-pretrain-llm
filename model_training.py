import torch
from transformers import (
    AutoModelForCausalLM,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

import datasets
from torch.utils.data import Dataset, Subset
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore")

EVAL_SIZE = 16  # evaluation dataset split size
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# Dataset Configs
# Here we need to overwrite the Dataset object to add interfaces to interact with the model
class CustomDataset(Dataset):
    def __init__(self, args, split="train"):
        """Initializes the custom dataset object."""
        self.args = args
        self.dataset = datasets.load_dataset(
            "parquet",
            data_files=args.dataset_name,
            split=split
        )

    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.dataset)

    def __getitem__(self, index):
        """Retrieves the specified data sample from the dataset by its index."""
        input_ids = torch.LongTensor(self.dataset[index]["input_ids"])
        labels = torch.LongTensor(self.dataset[index]["input_ids"])

        return {"input_ids": input_ids, "labels": labels}

# Training Configs
@dataclass
class CustomArguments(TrainingArguments):
    dataset_name: str = field(
        default="./data/packaged_pretrain_dataset.parquet")       # path / HF name the loader will open
    num_proc: int = field(default=1)                              # number of CPU workers used by datasets for map/tokenization
    max_seq_length: int = field(default=32)                       # hard truncation limit when creating input_ids

    # Core training configs
    seed: int = field(default=0)                                  # reproducible runs
    optim: str = field(default="adamw_torch")                     # adam with weight_decay
    max_steps: int = field(default=30)                            # Hard stop after max_steps
    per_device_training_batch_size: int = field(default=2)        # micro-batch that fits the GPU/CPU units, real_batch size = per_device_training_batch_size * num_of devices

    # Other training configs
    learning_rate: float = field(default=5e-5)                     # base lr feed to the optimiser
    weight_decay: float = field(default=0.0)                       # helps prevent overfitting, used only by AdamW
    warmup_steps: int = field(default=10)                          # linear LR ramp-up steps before reaching full LR, stabilises first updates
    lr_scheduler_type: str = field(default="linear")               # LR schedule once warm-up is done
    gradient_checkpointing: bool = field(default=True)             # forward activations are recomputed during backward
    dataloader_num_workers: int = field(default=2)                 # CPU workers for PyTorch Dataloader
    # bf16: bool = field(default=True)                             # doesn't work for MacOS
    gradient_accumulation_steps: int = field(default=1)            # Accumulate N micro-batches before calling optimiser.step()

    # Logging configs
    logging_steps: int = field(default=3)                          # log frequency
    report_to: str = field(default="none")                         # where to send logs

    # Saving configs
    save_strategy: str = field(default="steps")                    # when to checkpoint, steps = every save_steps
    save_steps: int = field(default=3)                             # the frequency for saving
    save_total_limit: int = field(default=2)                       # save at most 2 checkpoints

# define the logger
class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self.logs.append(logs)

    def __init__(self):
        self.logs = []

# ---------------------------------------------------------------------------
# 1. Training Preparation
# ---------------------------------------------------------------------------
# 1.A. Load model
model_path = "./models/TinySolar-308m-4k-init"
model_before = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
    use_cache=False,
)

# ---------------------------------------------------------------------------
# 2. Training Configs
# ---------------------------------------------------------------------------
parser = HfArgumentParser(CustomArguments)
args, = parser.parse_args_into_dataclasses(
    args=["--output_dir", "output"]
)
dataset = CustomDataset(args=args)
training_dataset = Subset(dataset, range(EVAL_SIZE, len(dataset)))

# ---------------------------------------------------------------------------
# 3. Run trainer and monitor loss
# ---------------------------------------------------------------------------
def main():
    # initialise the logger
    loss_logging_callback = LossLoggingCallback()

    trainer = Trainer(
        model=model_before,
        args=args,
        train_dataset=training_dataset,
        eval_dataset=None,
        callbacks=[loss_logging_callback],
    )

    trainer.train()

if __name__ == "__main__":
    main()


