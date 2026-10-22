# Kod: 4-tajriba: Skywork (tayyor) + RLOO

**A — hakam (shu tajribaga xos)**

| Qadam | Kod | Vazifa |
|---|---|---|
| A1 | [eval_judge.py:27](../../../rlhf/eval_judge.py#L27) | Hakamni 181 ta juftlikda tekshirish, o'qitishsiz |
| A2 | [judges.py:40](../../../rlhf/judges.py#L40) | Skywork'ni Hugging Face'dan yuklash |
| A3 | [judges.py:52](../../../rlhf/judges.py#L52) | **Skywork'ning o'z** tokenizeri bilan qayta tokenlash (Qwen3 ≠ Qwen2.5) |
| A4 | [judges.py:59](../../../rlhf/judges.py#L59) | Batch bilan baholash |
| B7 | [train_ppo.py:207](../../../rlhf/train_ppo.py#L207), [257-258](../../../rlhf/train_ppo.py#L257) | `--judge hf`; ballar har qadamda normallashtiriladi |
| B10 | [train_ppo.py:270-272](../../../rlhf/train_ppo.py#L270) | RLOO advantage (2-tajriba bilan bir xil) |

Kaggle notebook: [rlhf_skywork_kaggle.ipynb](rlhf_skywork_kaggle.ipynb)

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
