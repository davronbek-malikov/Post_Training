# 1. Reward model: base model (Qwen2.5-0.5B) — RLHF v1

| | |
|---|---|
| Reward model | Qwen2.5-0.5B **base** (hech qachon fine-tune qilinmagan) + LoRA + bitta son chiqaradigan head |
| O'qitilganmi? | Ha: reward model ham, keyin SFT model ham (PPO bilan) |
| Data | 339 + 18 juftlik (`dpo/data_v2`) |
| Natija | val acc **0.72**; test **40/68** (DPO v2: 44/68) |
| Muammo | "Qisqa javob yaxshi" deb o'rgangan: ball va uzunlik korrelyatsiyasi −0.63 |
| Kod | `rlhf/train_reward.py` (standart `--out 1_base-0.5b`), notebook `rlhf/rlhf_kaggle.ipynb` |

Bu papkada: `meta.json` (aniqlik, ball o'rtachasi va og'ishi), `training_log.json` (har qadamdagi loss).
Model og'irliklari (`adapter_model.safetensors`, `head.pt`) katta, shuning uchun git'da yo'q.
