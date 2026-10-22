# RLHF kodini qadamma-qadam o'rganish

Har bir tajriba uch qismdan iborat: **hakam (reward model)** tayyorlanadi → **RL tsikli** bizning SFT modelni
o'qitadi → **test**. RL tsikli (B qism) hamma tajribada bir xil; faqat hakam (A qism) va advantage hisoblash usuli
(PPO yoki RLOO) farq qiladi. Qator raqamlari shu fayl yozilgan paytdagi kod bo'yicha.

## Umumiy: RL tsikli (`train_ppo.py`) — hamma tajribada

| Qadam | Fayl va qator | Nima qiladi |
|---|---|---|
| B1 | [train_ppo.py:186](train_ppo.py#L186), funksiya [load_policy](train_ppo.py#L53) | **Policy**: base + SFT v3 LoRA, o'qitiladi (GPU 0) |
| B2 | [train_ppo.py:188](train_ppo.py#L188), [load_reference](train_ppo.py#L63) | **Reference**: o'sha SFT, muzlatilgan — KL uchun (GPU 1) |
| B3 | [train_ppo.py:217](train_ppo.py#L217) | 776 ta promptni o'qish (`dpo/data/prompts.jsonl`) |
| B4 | [train_ppo.py:232](train_ppo.py#L232), [rollout](train_ppo.py#L105) | Policy har promptga javob yozadi (sampling, temperature 0.7) |
| B5 | [train_ppo.py:242](train_ppo.py#L242), [logprobs](train_ppo.py#L75) | Har bir javob tokenining ehtimolligi: policy (`old`) |
| B6 | [train_ppo.py:244](train_ppo.py#L244) | Xuddi shu, reference model uchun (`ref`) |
| B7 | **tajribaga qarab** (quyida) | Hakam javobga ball beradi |
| B8 | [train_ppo.py:259](train_ppo.py#L259) | To'xtamagan javobga jarima (−1) |
| B9 | [train_ppo.py:260](train_ppo.py#L260) | **KL** = log π − log π_ref: model SFT'dan qancha uzoqlashgan |
| B10 | PPO: [265](train_ppo.py#L265) / RLOO: [270-272](train_ppo.py#L270) | **Advantage**: javob kutilganidan qancha yaxshi |
| B11 | [train_ppo.py:282-284](train_ppo.py#L282) | **PPO-clip yangilanish**: ratio = π_yangi/π_eski, [0.8, 1.2] da cheklanadi |
| B12 | [train_ppo.py:308](train_ppo.py#L308), [label.rule_score](../dpo/label.py#L162) | Hakamdan mustaqil **qoida bali**: reward hacking tekshiruvi |
| B13 | [train_ppo.py:327](train_ppo.py#L327) | Adapterni har 20 qadamda saqlash |
| B14 | [eval/run_eval.py:41](../eval/run_eval.py#L41), [auto_check](../eval/run_eval.py#L28) | 68 ta test savoli va avtomatik tekshiruv |

## 1-tajriba: base 0.5B reward model + PPO (RLHF v1)

| Qadam | Fayl va qator | Nima qiladi |
|---|---|---|
| A1 | [dpo/label.py:162](../dpo/label.py#L162), [score](../dpo/label.py#L155) | Qoidalar javobni baholaydi (arifmetika, ro'yxat, tarjima, rad etish) |
| A2 | `dpo/data_v2/train.jsonl` | DPO v2 juftliklari: 339 ta chosen/rejected |
| A3 | [train_reward.py:29](train_reward.py#L29) `encode`, [35](train_reward.py#L35) `load_pairs` | Prompt + javob → tokenlar (chat template) |
| A4 | [scalar_model.py:19](scalar_model.py#L19) `ScalarModel`, [45](scalar_model.py#L45) `build` | Til modeli + **bitta son chiqaradigan head** (lm_head o'rniga) |
| A5 | [scalar_model.py:92](scalar_model.py#L92) `last_scores` | Javobning **oxirgi tokenidagi** son = ball |
| A6 | [train_reward.py:143](train_reward.py#L143) | **Bradley-Terry loss**: −log σ(r_chosen − r_rejected) |
| A7 | [train_reward.py:46](train_reward.py#L46) `evaluate` | Val aniqlik: chosen > rejected ulushi |
| A8 | [scalar_model.py:59](scalar_model.py#L59) `save` | Reward model → `models/1_base-0.5b/` |
| B7 | [train_ppo.py:191](train_ppo.py#L191), [250](train_ppo.py#L250) | Reward model ball beradi |
| B7+ | [train_ppo.py:194](train_ppo.py#L194), [values](train_ppo.py#L83) | **Value model (critic)**: faqat PPO'da |
| B10 | [gae](train_ppo.py#L93), [train_ppo.py:265](train_ppo.py#L265) | **GAE** advantage (critic bilan) |
| B11+ | [train_ppo.py:295](train_ppo.py#L295) | Value model loss (critic ham o'qitiladi) |
| Kaggle | [models/1_base-0.5b/rlhf_kaggle.ipynb](models/1_base-0.5b/rlhf_kaggle.ipynb) | Hammasi ketma-ket |

## 2-tajriba: SFT nusxasi 1.5B reward model + RLOO (RLHF v2)

| Qadam | Fayl va qator | Nima qiladi |
|---|---|---|
| A1 | [build_rm_data.py:40](build_rm_data.py#L40) `prompt_pairs` | 8 ta javobning **barcha** juftliklari, ball farqi ≥ 0.5 ([51](build_rm_data.py#L51)) |
| A2 | [build_rm_data.py:32](build_rm_data.py#L32) `golds` | Namunaviy javob ham qo'shiladi (rad etishda uzun va qisqa) |
| A3 | [build_rm_data.py:65](build_rm_data.py#L65) `balance` | **Uzunlik muvozanati**: chosen qisqa = chosen uzun |
| A4 | [build_rm_data.py:89-90](build_rm_data.py#L89) | Val **promptlar** bo'yicha ajratiladi → 1 758 + 181 juftlik |
| A5 | [scalar_model.py:30](scalar_model.py#L30) `backbone_of` | Reward model **SFT v3 dan** boshlanadi (`--sft-adapter`) |
| A6 | [train_reward.py:143](train_reward.py#L143) | Bradley-Terry loss (1-tajriba bilan bir xil) |
| A7 | [train_reward.py:64](train_reward.py#L64) `length_gate` | **Uzunlik tekshiruvi**: hakam qisqa javobni ortiqcha yoqtiradimi |
| A8 | [train_reward.py:85](train_reward.py#L85) `length_slope`, [193-199](train_reward.py#L193) | **Length-controlled reward**: ball − koef × uzunlik |
| B7 | [train_ppo.py:181](train_ppo.py#L181) | Tekshiruvdan o'tmagan hakam bilan RL boshlanmaydi |
| B7 | [train_ppo.py:191](train_ppo.py#L191), [scalar_model.load](scalar_model.py#L66) | Reward model **float16**'da (GPU joyi) |
| B7 | [train_ppo.py:254](train_ppo.py#L254) | Uzunlik tuzatishi RL'da ham |
| B4+ | [train_ppo.py:230](train_ppo.py#L230) | **RLOO**: har prompt 4 marta (guruh) |
| B10 | [train_ppo.py:270-272](train_ppo.py#L270) | **RLOO advantage**: ball − qolgan 3 tasining o'rtachasi (critic yo'q) |
| Kaggle | [models/2_sft-1.5b/rlhf_v2_kaggle.ipynb](models/2_sft-1.5b/rlhf_v2_kaggle.ipynb) | Data → reward model → RLOO → test |

## 4-tajriba: tayyor Skywork reward model + RLOO

| Qadam | Fayl va qator | Nima qiladi |
|---|---|---|
| A1 | [eval_judge.py:27](eval_judge.py#L27) | Hakamni o'qitishsiz tekshirish: 181 val juftlikda aniqlik |
| A2 | [judges.py:40](judges.py#L40) `HFReward` | Skywork'ni Hugging Face'dan yuklaydi (o'qitish **yo'q**) |
| A3 | [judges.py:52](judges.py#L52) `text` | Skywork'ning **o'z** chat templati: javob matnga qaytarilib qayta tokenlanadi (Qwen3 ≠ Qwen2.5) |
| A4 | [judges.py:59](judges.py#L59) `score` | Batch bilan ball (logits[0]) |
| B7 | [train_ppo.py:207](train_ppo.py#L207), [257-258](train_ppo.py#L257) | `--judge hf`: Skywork ball beradi, har qadamda normallashtiriladi |
| B10 | [train_ppo.py:270-272](train_ppo.py#L270) | RLOO advantage (2-tajriba bilan bir xil) |
| Kaggle | [models/4_skywork-1.7b/rlhf_skywork_kaggle.ipynb](models/4_skywork-1.7b/rlhf_skywork_kaggle.ipynb) | Hakam testi → RLOO → test |

## 5-tajriba: katta LLM hakam (Gemini, API) + RLOO

| Qadam | Fayl va qator | Nima qiladi |
|---|---|---|
| A1 | [judges.py:31](judges.py#L31) `load_env` | Kalitni `.env` dan o'qiydi (Kaggle'da: Secrets) |
| A2 | [judges.py:84](judges.py#L84) `LLMJudge`, [98](judges.py#L98) `check` | Boshida kalit va model nomini tekshiradi; xato bo'lsa darhol to'xtaydi |
| A3 | [judges.py:68](judges.py#L68) `INSTRUCTIONS` | Hakamga ko'rsatma: nimani baholash, uzunlikni mukofotlamaslik, JSON javob |
| A4 | [judges.py:107](judges.py#L107) `ask`, [110-111](judges.py#L110) | Bitta so'rov = bitta promptning 4 ta javobi, **tartibi aralashtiriladi** (position bias) |
| A5 | [judges.py:119](judges.py#L119) | API so'rov; 429 (limit) bo'lsa kutib qayta urinadi |
| A6 | [judges.py:136](judges.py#L136) | Ballar asl tartibga qaytariladi |
| A7 | [judges.py:144](judges.py#L144) `score`, [147](judges.py#L147) | Promptlar parallel so'raladi (2 ta bir vaqtda) |
| A8 | [judges.py:155](judges.py#L155) | So'rov xato bo'lsa 4 ta javobga bir xil ball → RLOO undan o'rganmaydi |
| B7 | [train_ppo.py:207](train_ppo.py#L207), [257-258](train_ppo.py#L257) | `--judge api`: LLM ball beradi, har qadamda normallashtiriladi |
| B10 | [train_ppo.py:270-272](train_ppo.py#L270) | RLOO advantage |
| Kaggle | [models/5_frontier-llm/rlhf_llm_judge_kaggle.ipynb](models/5_frontier-llm/rlhf_llm_judge_kaggle.ipynb) | Kalit (Secrets) → hakam testi → RLOO → test |

## O'qish tartibi (tavsiya)

1. `dpo/label.py` — "yaxshi javob" nima ekanini qoidalar qanday aniqlaydi.
2. `scalar_model.py` + `train_reward.py` — reward model qanday yasaladi va o'qitiladi (1-tajriba).
3. `train_ppo.py` B1–B14 — RL tsikli: rollout → ball → KL → advantage → yangilanish.
4. `build_rm_data.py` va `train_reward.py` dagi uzunlik tekshiruvi — v1 xatolari qanday tuzatilgan (2-tajriba).
5. `judges.py` — tayyor hakamlar (4 va 5-tajriba).
6. Har bir papkadagi `RESULTS.md` — natija va uni qanday o'qish.
