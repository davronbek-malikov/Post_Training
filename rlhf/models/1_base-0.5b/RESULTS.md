# Natijalar: Base model (Qwen2.5-0.5B) reward model + PPO — RLHF v1

Test: 68 ta savol (`eval/eval_set.json`), avtomatik tekshiruv. Hamma tajriba SFT v3 dan boshlangan.

| Vazifa | SFT v3 (boshlanish) | **Bu tajriba** | DPO v2 (eng yaxshi) |
|---|---|---|---|
| Fakt | 6/13 | **7/13** | 7/13 |
| Tushuntirish | 3/10 | **2/10** | 3/10 |
| Ro'yxat | 10/10 | **9/10** | 10/10 |
| Tarjima | 3/10 | **2/10** | 3/10 |
| Arifmetika | 8/12 | **8/12** | 8/12 |
| "Bilmayman" | 4/5 | **4/5** | 5/5 |
| Zararli so'rovni rad etish | 4/4 | **4/4** | 4/4 |
| Imtihonda ko'chirishni rad etish | 3/4 | **4/4** | 4/4 |
| **Jami** | **41/68** | **40/68** | **44/68** |

## RL jarayoni

| Ko'rsatkich | Boshida (10% qadam) | Oxirida (10% qadam) |
|---|---|---|
| Hakam bali | 25.87 | 26.39 |
| Qoida bali (hakamdan mustaqil tekshiruv) | 0.72 | 0.65 |
| KL (SFT'dan uzoqlashish) | 0.05 | 1.65 |
| Javob uzunligi (token) | 30.08 | 26.96 |

- Qadamlar: 97; RL paytida yozilgan javoblar: 776
- Hakam bali va qoida bali korrelyatsiyasi: +0.08
- Hakam bali va javob uzunligi korrelyatsiyasi: -0.63
- Test javoblarining o'rtacha uzunligi: 61 belgi

## Fayllar (`results/`)

| Fayl | Nima |
|---|---|
| `test_answers.json` | 68 ta test savoli, model javobi, to'g'ri javob va `auto_pass` |
| `rl_training_log.json` | RL'ning har qadami: hakam bali, qoida bali, KL, uzunlik, loss |
| `rollouts.jsonl` | RL paytida yozilgan har bir javob, uning hakam bali (`rm`) va qoida bali (`rule`) |
