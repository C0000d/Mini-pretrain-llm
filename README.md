# Lightweight Pretrain pipeline on Llama-based model  
Developed a CLI-based pipeline for pretraining LLaMA-style language models on diverse, code-enriched web text, enabling flexible hyperparameter tuning and measurable learning progress under constrained hardware (laptop-scale).

---

## What this repo shows

| Stage                      | What I actually implemented                                                                                                                                                                                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data collection**        | Pulled 60k RedPajama subset (`upstage/Pretraining_Dataset`) + 9 Python scripts.                                                                                                                                                                  |
| **Quality filtering**      | • paragraph‑length<br>• duplicate‑paragraph<br>• global de‑duplication<br>• FastText L2 model to keep **English > 0.4prob** only.                                                                                                                |
| **Tokenisation & packing** | • Applied HF `AutoTokenizer`<br>• Added EOS/BOS<br>• Flattened & packed into **fixed 32‑token blocks**, saved as int32 Parquet.                                                                                                                  |
| **Model preparation**      | Explored among 4 ways of preparing the model: <br>• Fresh 12‑layer Llama config (random weights)<br>• Load *TinySolar‑248M* (12L) checkpoint<br>• **Down‑scale** to 10L<br>• **Up‑scale** to 16L: copy bottom‑8 + top‑8 layers.(Decided on this) |
| **Trainer set‑up**         | • Custom `Dataset` wrapper<br>• AdamW‑torch<br>• gradient‑checkpointing<br>• 32‑token micro‑batches<br>• **30steps** hard stop<br>• CPU‑only run.                                                                                                |
| **Monitoring**             | Callback that captures `loss` every 3 steps.                                                                                                                                                                                                     |
| **Evaluation**             | Benchmark preplexity on model before & after pretraining to verify the uncertainty of model producing response.                                                                                                                                  |

---

## Models Used

This repo uses instruction-tuned TinySolar models provided by [Upstage](https://github.com/upstage-open/solartts), such as:

- `TinySolar-248m-4k`
- `TinySolar-308m-4k-init`

> ⚠️ Models are not included in this repo due to the size constraints.  

---

## Dataset

We train on the **Upstage/Pretraining_Dataset** dataset, a subset of the RedPajamas. Along with the code scraping from github.

| Field       | Description                                     |
|-------------|-------------------------------------------------|
| Type        | Raw Text                       |
| Source      | Hugging Face Datasets & GitHub                  |
| Preprocess  | Tokenization, truncation, formatting, packaging |
| Target Task | Pretraining                                     |

---

## Quick run

> **Prereqs**: Python 3.10+, `torch`, `transformers>=4.40`, `datasets`, `fasttext`, `parquet`.

```bash
# 1. Clean & package the corpus
python data_preparation.py
python data_packaging.py

# 2. Build a 16‑layer TinySolar skeleton & load hybrid weights
python model_preparation.py      

# 3. Fire one tiny pre‑train sweep
python model_training.py      

# 4. Benchmark evaluation
python model_evaluation.py
```

---

## Key techniques demonstrated
- Heuristic text hygiene – length, duplication & language filters that together discarded ~40% noisy lines before tokenisation.
- Vector‑friendly packing – converts ragged examples to fixed‑length 32‑token blocks.
- Layer management – shows both down‑scaling (strip mid‑layers) and up‑scaling (clone bottom/top) without re‑training from scratch.
- HF Trainer on Parquet – custom Dataset + dataclass args for quick sweep.
- Memory tricks – bfloat16, gradient‑checkpointing, int32storage so the whole pipeline fits into a laptop RAM budget.
- Evaluation on perplexity, showing 20% reduction on the uncertainty of model producing responses after the pretraining.
---

## Issues & Limitations
- CPU‑only run — metrics & generations are qualitative.
- 16‑layer model is 308 M params – still orders of magnitude smaller than production LLMs.
- Large models (e.g. `.bin`, `.safetensors`) are not pushed to GitHub due to the 100MB limit.
- `L2_language_model.bin`, `TinySolar-248m-4k`, `TinySolar-308m-4k-init` is excluded and must be manually downloaded if used.
- Current code assumes model directory structure from original TinySolar release. Adjust paths if using HF-hosted models.

