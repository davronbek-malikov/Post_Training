# Kod: 1-tajriba: base 0.5B + PPO

**A — hakam (shu tajribaga xos)**

| Qadam | Kod | Vazifa |
|---|---|---|
| A1 | [label.py:162](../../../dpo/label.py#L162) | Qoidalar javoblarni baholaydi (matematika, ro'yxat, rad etish...) |
| A2 | [train.jsonl](../../../dpo/data_v2/train.jsonl) | 339 ta chosen/rejected juftlik |
| A3 | [scalar_model.py:19](../../../rlhf/scalar_model.py#L19), [45](../../../rlhf/scalar_model.py#L45) | LLM + matn o'rniga **bitta son chiqaradigan head** |
| A4 | [scalar_model.py:92](../../../rlhf/scalar_model.py#L92) | Ball javobning **oxirgi tokenida** o'qiladi |
| A5 | [train_reward.py:143](../../../rlhf/train_reward.py#L143) | **Loss**: −log σ(r_chosen − r_rejected) |
| B7 | [train_ppo.py:250](../../../rlhf/train_ppo.py#L250) | Reward model javoblarga ball beradi |
| B7+ | [train_ppo.py:194](../../../rlhf/train_ppo.py#L194) | **Value model (critic)**, faqat PPO'da |
| B10 | [gae](../../../rlhf/train_ppo.py#L93) | Critic yordamida **GAE** advantage |

Kaggle notebook: [rlhf_kaggle.ipynb](rlhf_kaggle.ipynb)

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
