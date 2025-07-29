# Mini Pretrain pipeline on Llama-based model  
*From raw web‑text ➜ cleaned Parquet ➜ token‑packed shards ➜ 16‑layer Llama‑family
checkpoint ➜ HF `Trainer` loop*

---

## What this repo shows

| Stage                      | What I actually implemented                                                                                                                                                                                                                                                                        |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Data collection**        | Pulled 60k RedPajama pages (`upstage/Pretraining_Dataset`) + 9 popular OSS Python files.                                                                                                                                                                                                           |
| **Quality filtering**      | • paragraph‑length guard<br>• duplicate‑paragraph detector<br>• global de‑dup pass<br>• FastText L2 model to keep **English > 0.4prob** only.                                                                                                                                                      |
| **Tokenisation & packing** | HF `AutoTokenizer` (Solar‑10.7B) → adds `<s> … </s>` → flattens & chops into **fixed 32‑token blocks**, saved as int32 Parquet.                                                                                                                                                                    |
| **Model preparation**      | In this section I tried 4 ways of preparing the model, and selected the 4th: <br>1) fresh 12‑layer Llama config (random weights)<br>2) load *TinySolar‑248M* (12L) checkpoint<br>3) **down‑scale** to 10L<br>4) **up‑scale** to 16L: copy bottom‑8 + top‑8 layers, reuse embed‑tokens & `lm_head`. |
| **Trainer set‑up**         | Custom `Dataset` wrapper, AdamW‑torch, gradient‑checkpointing, 32‑token micro‑batches, **30steps** hard stop, CPU‑only run.                                                                                                                                                                        |
| **Monitoring**             | Minimal callback that captures `loss` every 3 steps.                                                                                                                                                                                                                                               |
| **Inference demo**         | Greedy & sampling generation from `checkpoint‑30` to verify the pipeline.                                                                                                                                                                                                                          |

---

## Models Used

This repo uses instruction-tuned TinySolar models provided by [Upstage](https://github.com/upstage-open/solartts), such as:

- `TinySolar-248m-4k`
- `TinySolar-308m-4k-init`

> ⚠️ Models are not included in this repo due to size constraints.  

---

## Dataset

We train on the **Upstage/Pretraining_Dataset** dataset, a subset of the Red Pajamas.

| Field       | Description                                     |
|-------------|-------------------------------------------------|
| Type        | Raw Text                                        |
| Source      | Hugging Face Datasets                           |
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
```

---

## Key techniques demonstrated
- Heuristic text hygiene – length, duplication & language filters that together discarded ~40% noisy lines before tokenisation.
- Vector‑friendly packing – converts ragged examples to fixed‑length 32‑token blocks.
- Layer management – shows both down‑scaling (strip mid‑layers) and up‑scaling (clone bottom/top) without re‑training from scratch.
- HF Trainer on Parquet – custom Dataset + dataclass args for quick sweep.
- Memory tricks – bfloat16, gradient‑checkpointing, int32storage so the whole pipeline fits into a laptop RAM budget.

---

## Issues & Limitations
- CPU‑only run; metrics & generations are qualitative.
- 16‑layer model is 308 M params – still orders of magnitude smaller than production LLMs.
- No evaluation harness (Rouge/BLEU) included; focus is on the pre‑training flow, not downstream task quality.
- Large models (e.g. `.bin`, `.safetensors`) are not pushed to GitHub due to the 100MB limit.
- `L2_language_model.bin`, `TinySolar-248m-4k`, `TinySolar-308m-4k-init` is excluded and must be manually downloaded if used.
- Current code assumes model directory structure from original TinySolar release. Adjust paths if using HF-hosted models.

