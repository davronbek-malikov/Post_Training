# 2. Reward model: SFT modelimizning nusxasi (Qwen2.5-1.5B + SFT v3) — RLHF v2

| | |
|---|---|
| Reward model | Qwen2.5-1.5B + **SFT v3** LoRA (vazifalarni allaqachon biladi) + reward LoRA + head |
| O'qitilganmi? | Ha: reward model ham, keyin SFT model ham (RLOO bilan) |
| Data | 1 758 + 181 juftlik (`rlhf/data_v2`, `rlhf/build_rm_data.py`), uzunlik muvozanatlangan |
| Natija | val acc **0.84**, uzunlik tekshiruvi o'tdi (tuzatishdan keyin pair_corr −0.09); test **39/68** (v1: 40, DPO v2: 44) |
| Kod | `train_reward.py --sft-adapter sft-lora-v3 --out 2_sft-1.5b`, notebook: shu papkadagi `rlhf_v2_kaggle.ipynb` |

Bu papkada: `meta.json` (aniqlik, uzunlik tuzatishi), `training_log.json`. Tahlil: `rlhf/README.md`, "Results: RLHF v2".
