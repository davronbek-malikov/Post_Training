# Kod: 2-tajriba: SFT nusxasi 1.5B + RLOO

**A — hakam (shu tajribaga xos)**

| Qadam | Kod | Vazifa |
|---|---|---|
| A1 | [build_rm_data.py:40](../../../rlhf/build_rm_data.py#L40) | 8 ta javobdan **barcha** juftliklar (1 939 juftlik) |
| A2 | [build_rm_data.py:65](../../../rlhf/build_rm_data.py#L65) | **Uzunlik muvozanati** |
| A3 | [scalar_model.py:30](../../../rlhf/scalar_model.py#L30) | Hakam **SFT v3 dan** boshlanadi |
| A4 | [train_reward.py:64](../../../rlhf/train_reward.py#L64) | **Uzunlik tekshiruvi** |
| A5 | [train_reward.py:85](../../../rlhf/train_reward.py#L85) | **Uzunlik tuzatishi**: ball − koeffitsient × uzunlik |
| B7 | [train_ppo.py:181](../../../rlhf/train_ppo.py#L181), [254](../../../rlhf/train_ppo.py#L254) | Xolis bo'lmagan hakam bilan RL boshlanmaydi; tuzatish RL'da ham qo'llanadi |
| B4+ | [train_ppo.py:230](../../../rlhf/train_ppo.py#L230) | **RLOO**: har bir savol 4 marta |
| B10 | [train_ppo.py:270-272](../../../rlhf/train_ppo.py#L270) | Advantage = ball − qolgan 3 tasining o'rtachasi |

Data: [train.jsonl](../../../rlhf/data_v2/train.jsonl), [val.jsonl](../../../rlhf/data_v2/val.jsonl). Kaggle notebook: [rlhf_v2_kaggle.ipynb](rlhf_v2_kaggle.ipynb)

## Umumiy RL tsikli (hamma tajribada bir xil, `train_ppo.py`)

| Qadam | Kod | Vazifa |
|---|---|---|
| B1 | [train_ppo.py:186](../../../rlhf/train_ppo.py#L186) | **Policy** (SFT v3) yuklanadi — o'rganadigan model |
| B2 | [train_ppo.py:188](../../../rlhf/train_ppo.py#L188) | **Reference** (SFT'ning muzlatilgan nusxasi) — KL penalty uchun |
| B4 | [train_ppo.py:232](../../../rlhf/train_ppo.py#L232) → [rollout](../../../rlhf/train_ppo.py#L105) | Policy javob yozadi |
| B5–B6 | [train_ppo.py:242-244](../../../rlhf/train_ppo.py#L242) | Policy va reference'da tokenlar ehtimolligi |
| B9 | [train_ppo.py:260](../../../rlhf/train_ppo.py#L260) | **KL**: model SFT'dan qancha uzoqlashdi |
| B11 | [train_ppo.py:282-284](../../../rlhf/train_ppo.py#L282) | **PPO-clip yangilanish**: har qadamda modelni ozgina o'zgartirish |
| B12 | [train_ppo.py:308](../../../rlhf/train_ppo.py#L308) | **Qoida bali**: reward hacking'ni mustaqil tekshirish |
| B13 | [train_ppo.py:327](../../../rlhf/train_ppo.py#L327) | Adapterni saqlash |
| B14 | [run_eval.py:41](../../../eval/run_eval.py#L41) | 68 savollik test |

To'liq qo'llanma (hamma tajriba bir joyda): [rlhf/CODE_GUIDE.md](../../CODE_GUIDE.md). Natijalar: [RESULTS.md](RESULTS.md).

Kod fayllari `rlhf/` va `dpo/` papkalarida qoladi: ular bir nechta tajribada umumiy ishlatiladi (masalan,
`train_ppo.py` — 1, 2, 4 va 5-tajribada). Havolalar o'sha fayllarning kerakli qatorini ochadi. Qator raqamlari
kodning hozirgi holatiga mos; kod o'zgarsa, biroz siljishi mumkin.
