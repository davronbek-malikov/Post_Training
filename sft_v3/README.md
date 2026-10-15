# SFT v3: new data, Qwen2.5-1.5B, LoRA and QLoRA (Kaggle GPU)

Same model and training script as v2 (`sft_v2/train.py`). Only the data changes, so the v2 → v3
difference shows the effect of the data.

| | |
|---|---|
| Build the data | `python sft_v3/build_v3.py` → `sft_v3/data/train.jsonl`, `val.jsonl` |
| New examples | `new_samples.txt` (165 kept after cleaning) |
| Notebook | `sft_v3_kaggle.ipynb` (import it on Kaggle, set Accelerator to GPU and Internet to On) |
| Settings | 3 epochs (v2 overfit after epoch 2), rank 64, batch size 8, learning rate 2e-4; best epoch saved |
| Adapters | `chat_ui/adapters/sft-lora-v3/`, `chat_ui/adapters/sft-qlora-v3/` (not in git) |
| Test answers | `eval/results/sft-lora-v3.json`, `sft-qlora-v3.json` |

## What changed in the data

| | v1 / v2 | v3 |
|---|---|---|
| Examples (train + val) | 1,252 | 1,583 |
| Alpaca-uz (machine-translated) | 49 | removed |
| General knowledge and code (`extra_general.txt`) | 42 | removed (not in scope, MODEL_SPEC.md §7) |
| Arithmetic | 100 | 200 |
| Translation | 160 | 317 (80 new pairs, many with opposite adjectives) |
| New course concepts (gradient clipping, KV cache, top-p, MoE…) | — | ~70, in Uzbek and English |
| Lists of practical tips | 64 | 95 (against refusals leaking into lists) |
| Exam cheating with a real question inside | 20 | 33 |
| Exam preparation (should be answered) | 12 | 25 |
| Test-question check | exact match | exact match + near-identical (similarity ≥ 0.98) |

New harmful-request examples were not added; the 27 refusals from `seeds.txt` are kept.
Refusals and "Bilmayman" make up 7% of the data.

The validation set is new, so v3 validation loss cannot be compared with v2; compare the test scores.
