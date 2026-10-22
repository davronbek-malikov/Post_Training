# 4. Reward model: tayyor model (Skywork/Skywork-Reward-V2-Qwen3-1.7B)

| | |
|---|---|
| Reward model | Hugging Face'dagi tayyor reward model, millionlab (asosan inglizcha) juftlikda o'qitilgan |
| O'qitilganmi? | Reward model — **yo'q** (tayyor). SFT model — **ha** (RLOO bilan) |
| Bizning 181 val juftlikda | aniqlik **0.70** (bizning v2: 0.84). Fakt 0.94, arifmetika 0.85, tushuntirish 0.86 — bizdan yaxshi; imtihonda ko'chirish 0.33, zararli 0.38, "Bilmayman" 0.58 — bizdan yomon |
| Uzunlik | deyarli bog'liq emas (pair_corr −0.14) |
| Natija (Kaggle, 59 daq) | test **40/68**: "Bilmayman" 5/5, zararli 4/4, tushuntirish 3/10; tarjima 1/10. Qoida bali 0.67 -> 0.65 |
| Kod | `rlhf/judges.py` (`HFReward`), `train_ppo.py --judge hf`, notebook: shu papkadagi `rlhf_skywork_kaggle.ipynb` |

Model o'qitilmagani uchun bu papkada og'irliklar yo'q: u har safar Hugging Face'dan yuklab olinadi.
