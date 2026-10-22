# Kod: 5-tajriba: katta LLM (Gemini) + RLOO

**A — hakam (shu tajribaga xos)**

| Qadam | Kod | Vazifa |
|---|---|---|
| A1 | [judges.py:31](../../../rlhf/judges.py#L31) | Kalitni `.env` dan o'qish (Kaggle'da: Secrets) |
| A2 | [judges.py:98](../../../rlhf/judges.py#L98) | Boshida kalitni tekshirish; noto'g'ri bo'lsa darhol to'xtash |
| A3 | [judges.py:68](../../../rlhf/judges.py#L68) | Hakamga **ko'rsatma** (uzunlikni mukofotlamaslik, JSON'da javob berish) |
| A4 | [judges.py:110-111](../../../rlhf/judges.py#L110) | Bitta so'rovda 4 ta javob, **aralashtirilgan tartibda** |
| A5 | [judges.py:119](../../../rlhf/judges.py#L119) | API so'rovi; 429 xatosidan keyin kutib, qayta urinish |
| A6 | [judges.py:136](../../../rlhf/judges.py#L136) | Ballarni asl tartibga qaytarish |
| A7 | [judges.py:155](../../../rlhf/judges.py#L155) | Xato so'rov → bir xil ballar, shuning uchun undan hech narsa o'rganilmaydi |
| B7 | [train_ppo.py:257-258](../../../rlhf/train_ppo.py#L257) | `--judge api` |
| B10 | [train_ppo.py:270-272](../../../rlhf/train_ppo.py#L270) | RLOO advantage |

Hakam testi: [eval_judge.py:27](../../../rlhf/eval_judge.py#L27) (`--judge api`). Kaggle notebook: [rlhf_llm_judge_kaggle.ipynb](rlhf_llm_judge_kaggle.ipynb)

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
