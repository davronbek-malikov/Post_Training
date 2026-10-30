# Post-Training Lesson — Full Summary (context file for a new Claude Code session)

> Read this first in a new session. It records everything done in the post-training lesson of the course,
> with numbers, file paths, decisions, mistakes and their fixes. Written on 2026-10-24, when the
> post-training work was finished and fully pushed to GitHub (`main` in sync with `origin/main`).

---

## 0. Context: who, what, why

- **Owner / teacher:** Davronbek (git user `Davronbek`). Teaches an LLM course to **Uzbek students**.
  Explanations for students should be beginner-level; in Uzbek when asked (Latin script).
- **Course structure:**
  1. Months 1–2: Deep Learning with neural networks.
  2. Months 3–4: **Pretraining** (tokenizer, Transformer, training from scratch). Taught already, not in this repo.
  3. Month 5 (this repo): **Post-training**: base model → SFT → DPO → RLHF → deployment.
  4. Next target (being planned): job-ready skills for **LLM Engineer / RAG Engineer** roles (the teacher looked at
     Saramin, the Korean job site). A RAG plan already exists: [docs/plans/2026-10-15-ustoz-1.5b-rag.md](docs/plans/2026-10-15-ustoz-1.5b-rag.md).
- **The product:** **TeachHelp** (earlier name "Ustoz"), a short-answer assistant for students of the course.
  Behaviour is defined in [MODEL_SPEC.md](MODEL_SPEC.md).
- **The repo is also the worked example for the course:** every stage is kept, measured on the same test and
  explained (READMEs, `CODE.md`, study-note comments in the scripts, partly in Uzbek).

### Environment
- Windows 10 laptop, **no GPU** (CPU only). Python scripts run locally for data building, eval and the chat UI.
- **Kaggle free GPU** (T4 / T4 x2, 16 GB) for all 1.5B training: each stage has a `*_kaggle.ipynb` notebook.
  Import → Accelerator GPU → Internet On → upload needed adapters as a Kaggle dataset → Run All.
- Repo path: `C:\Users\davro\OneDrive\Desktop\Post-Training`. GitHub remote, branch `main`.
- **Not in git** (see `.gitignore`): trained adapters `chat_ui/adapters/`, `Qwen2.5-1.5B/` base model (~3 GB),
  reward-model weights (`rlhf/models/**/*.safetensors`, `head.pt`), `.env` (API keys), `*.zip`, `sft/external/`,
  `deploy/vercel/teach-help/`. `Qwen2.5-0.5B/` **is** in git.
- Local adapters present: `sft-lora`, `sft-qlora`, `sft-lora-1.5b`, `sft-qlora-1.5b`, `sft-lora-v3`, `sft-qlora-v3`,
  `sft-lora-dpo`, `sft-lora-dpo-v2`, `sft-lora-rlhf`, `sft-lora-rlhf-v2`, `sft-lora-rlhf-skywork`, `sft-lora-rlhf-llm`.
- Commit style: conventional prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `results:`, `deploy:`).

---

## 1. Model spec (the target behaviour) — [MODEL_SPEC.md](MODEL_SPEC.md)

Written **before** any training (v0.1 2026-10-11, v0.2 2026-10-12). Training data follows it; every model is judged by it.

- **Language:** Uzbek (Latin) by default; English if asked in English; no mixing except translation; course terms
  (fine-tuning, token, embedding, RAG) stay in English.
- **Style:** 1–5 sentences or a short numbered list, beginner level, no repeating the question, no invented web text,
  no continuing the conversation, stop when done.
- **Tasks:** (1) facts about course topics, (2) short concept explanation, (3) list of tips/steps with exact count,
  (4) Uzbek↔English sentence translation, (5) basic arithmetic with correct order of operations.
- **Limits:** say **"Bilmayman"** ("I don't know") instead of inventing; politely refuse harmful requests;
  refuse **exam cheating** (but explaining a topic / practice question is fine).
- **Out of scope v0.x:** code, multi-step math, general knowledge (geography/history), long essays, other languages.

---

## 2. Evaluation first — [eval/](eval/)

- [eval/eval_set.json](eval/eval_set.json): **68 held-out questions**, 8 task types:
  Factual 13, Explanation 10, List 10, Translation 10, Arithmetic 12, "Bilmayman" 5, Refuses harmful 4, Refuses cheating 4.
- [eval/run_eval.py](eval/run_eval.py): `python eval/run_eval.py --base Qwen2.5-1.5B --model <adapter> [--name] [--limit] [--max-new-tokens 100]`
  → `eval/results/<name>.json`.
- [eval/summarize.py](eval/summarize.py) → [results/per_task_scores.md](results/per_task_scores.md) and `.csv`.
- **Automatic check:** pass = expected keyword present, or the requested number of list items.
  It does **not** check language or clean ending. On 68 questions a **1–3 point difference is noise**.
- Training data is checked against test questions (exact match; from v3 also near-duplicates, similarity ≥ 0.98).

---

## 3. Stage 0 — Base model + local chat UI

- Qwen2.5-0.5B runs on the laptop CPU with a small chat UI ([chat_ui/](chat_ui/)).
- **Score 5/68.** A base model only continues text: no answering format, no stopping, no refusing.
  This is the problem post-training solves. (Base 1.5B later: 15/68.)

---

## 4. Stage 1 — SFT v1 (Qwen2.5-0.5B): teach the format — [sft/](sft/)

- **Data** built by [sft/build_sft.py](sft/build_sft.py): ~1,252 examples (train + val) from hand-written seeds
  ([sft/seeds.txt](sft/seeds.txt), incl. 27 refusals), 500 Uzbek samples, translated Alpaca (`external_alpaca_uz.txt`),
  OASST English (`external_oasst_en.txt`), `extra_general.txt`. Downloads via `sft/fetch_external.py`.
- **Important fix:** training answers must end with `<|endoftext|>` or the model never learns to stop.
- **Two ways to train the same LoRA adapter:**

| | LoRA on CPU ([sft/lora_cpu/](sft/lora_cpu/)) | QLoRA on Kaggle ([sft/qlora_kaggle/](sft/qlora_kaggle/)) |
|---|---|---|
| Base precision | full, frozen | 4-bit, frozen |
| Settings | 3 epochs, rank 16, batch 8, lr 2e-4 | 10 epochs, rank 64, batch 8, lr 2e-4, best epoch saved |
| Trainable params | 8.8M of 503M (1.75%) | — |
| Time | 151 min on laptop CPU | Kaggle GPU |
| Val loss | 2.09, 1.69, 1.62 | 2.00, 1.56, **1.48** (best), then rising to 1.91 (overfit) |
| **Test** | **32/68** | **32/68** |

- QLoRA cannot run on CPU (bitsandbytes needs a GPU). The two runs had different settings, so they compare setups,
  not LoRA vs QLoRA alone. (Colab was tried first, then dropped for Kaggle.)

---

## 5. Stage 2 — SFT v2 (Qwen2.5-1.5B): a bigger model — [sft_v2/](sft_v2/)

- Same data, 3× bigger model, **LoRA and QLoRA with identical settings** (`sft_v2/train.py --method lora|qlora`).
- 5 epochs, rank 64, batch 8, lr 2e-4, best epoch saved. LoRA uses 32-bit base + gradient checkpointing to fit 16 GB.

| | LoRA 1.5B | QLoRA 1.5B |
|---|---|---|
| Val loss | 1.44, **1.19**, 1.23, 1.35, 1.46 | 1.49, **1.22**, 1.23, 1.37, 1.46 |
| **Test** (base 1.5B: 15/68) | **44/68** | 42/68 |

- **Lessons:** model size helped most; 4-bit QLoRA cost only 2 points; both overfit after epoch 2 → 2–3 epochs enough.

---

## 6. Stage 3 — SFT v3: better data — [sft_v3/](sft_v3/)

- Same model and script as v2; **only the data changed** ([sft_v3/build_v3.py](sft_v3/build_v3.py), [sft_v3/new_samples.txt](sft_v3/new_samples.txt)).
- 1,583 examples. Removed machine-translated Alpaca-uz (49) and off-topic general/code (42). Arithmetic 100 → 200,
  translation 160 → 317, ~70 new course concepts (gradient clipping, KV cache, top-p, MoE…), lists 64 → 95,
  exam cheating 20 → 33, exam preparation (must be answered) 12 → 25. Refusals + "Bilmayman" = 7% of data.
- 3 epochs. Val loss LoRA 1.35, **1.08**, 1.14; QLoRA 1.39, **1.11**, 1.18 (new val set, not comparable to v2).
- **Test 41/68 (LoRA and QLoRA).** Total did not move, but refusal text ("Kechirasiz") no longer leaks into normal
  answers (4 → 1/0), and questions asked *during* an exam are refused. Still wrong: facts, "Bugun havo issiq" → "cold",
  some garbled refusals.
- **SFT v3 LoRA became the starting point for DPO and all RLHF runs** because its behaviour is cleanest.

---

## 7. Stage 4 — DPO: learning from preferences — [dpo/](dpo/)

Pipeline (all from step 2 in `dpo_kaggle.ipynb`; needs `sft_v3_adapters.zip` uploaded as Kaggle dataset):

| Step | Script | Output |
|---|---|---|
| 1. Prompts | [dpo/build_prompts.py](dpo/build_prompts.py) | `data/prompts.jsonl`: 776 prompts (SFT v3 data + 100 new arithmetic) |
| 2. Answers | [dpo/generate.py](dpo/generate.py) | `data/generations.jsonl`: N sampled answers per prompt |
| 3. Pairs | [dpo/label.py](dpo/label.py) | `data/train.jsonl`, `val.jsonl`: chosen / rejected by **rules** |
| 4. Train | [dpo/train_dpo.py](dpo/train_dpo.py) | `chat_ui/adapters/sft-lora-dpo[-v2]/` |

- **Labelling rules (label.py):** refusals must contain "yordam bera olmayman"/"can't help" and be short; "Bilmayman"
  must start the answer; arithmetic checks every `a op b = c`; lists need exact item count; translation needs every
  word of the reference (word overlap alone accepts "cold" for "hot"); others use word overlap with the reference.
  Penalties −0.5: refusal text in a normal answer, no stop, repetition, too long, wrong language.
- **Training:** policy = base + SFT v3 adapter (trainable); reference = same model frozen, log-probs precomputed once
  (only one model in memory). Loss `-log σ(β·[(logπ−logπ_ref)(chosen) − (logπ−logπ_ref)(rejected)])`, β = 0.1.
  Sanity check: val loss at start = **0.693 (ln 2)**.

**DPO v1** (313 pairs, 4 answers/prompt, lr 2e-5, 2 epochs): val loss 0.693 → 0.311, acc 0.75. **Test 40/68.**
Problems: **length bias** (chosen shorter in 63% of pairs → "shorter is better", answers 87 → 51 chars), chosen reward
also fell (−0.7), garbled refusals chosen, a bug marking Uzbek answers to digit-only prompts as "wrong language".

**DPO v2 fixes** (`dpo_v2_kaggle.ipynb`, data in `dpo/data_v2/`):
- Refusals: chosen = one fixed clean sentence; "Bilmayman": SFT reference answer.
- Rejected = bad answer closest in length to chosen; skip pairs with length ratio outside 0.6–1.7.
- **SFT loss on chosen added** (`--sft-weight 1.0`); best epoch by DPO + SFT loss.
- lr 5e-6; 8 answers per prompt; list items need ≥ 2 words; language check skipped for digit-only prompts;
  near-identical pairs skipped (except arithmetic/translation).

**DPO v2 result:** 357 pairs (chosen shorter 47%), val loss 0.693 → 0.571, acc 0.78, chosen reward +0.22.
**Test 44/68 — the FINAL MODEL.** Harmful 4/4, cheating 4/4, "Bilmayman" 5/5, no stray "Kechirasiz", avg length 76.
Changed 3 test answers vs SFT v3, all for the better. Not fixed (as expected from DPO): wrong facts, "hot" → "cold".

---

## 8. Stage 5 — RLHF: reward model + RL — [rlhf/](rlhf/)

Classic InstructGPT recipe, compared with DPO on the same pairs and the same test.
Code: [rlhf/scalar_model.py](rlhf/scalar_model.py) (transformer + scalar head, for reward and value models),
[rlhf/train_reward.py](rlhf/train_reward.py) (Bradley-Terry `-log σ(r_chosen − r_rejected)`, val acc, length-bias report,
score normalisation), [rlhf/train_ppo.py](rlhf/train_ppo.py) (PPO and `--algo rloo`), [rlhf/build_rm_data.py](rlhf/build_rm_data.py),
[rlhf/judges.py](rlhf/judges.py) (HF reward model / API LLM judge), [rlhf/eval_judge.py](rlhf/eval_judge.py),
[rlhf/collect_results.py](rlhf/collect_results.py). Step-by-step code tables: [rlhf/CODE_GUIDE.md](rlhf/CODE_GUIDE.md).
One folder per experiment in [rlhf/models/](rlhf/models/) with `ABOUT.md`, `CODE.md`, `RESULTS.md`, notebook, results.

**PPO loop (per batch of 8 prompts):** rollout (temp 0.7) → reward model score (−1 if no stop) → per-token KL penalty
`−β(logπ − logπ_ref)` (β 0.05 in v1, 0.1 in v2) → GAE advantages (γ 1, λ 0.95) → 2 PPO epochs, clipped ratio ε 0.2,
clipped value loss. Four models in memory on T4 x2: policy, reference, reward, value (critic).
**Reward-hacking check:** every step also logs the rule score from `dpo/label.py` (never seen by RL). RM score up while
rule score down = the policy is fooling the judge. Sanity: KL = 0 at step 1.

| # | Folder | Reward model (judge) | RL | Judge val acc | Test |
|---|---|---|---|---|---|
| 1 | `1_base-0.5b` | Qwen2.5-0.5B base, trained on 339 pairs | PPO | 0.72 (18 pairs) | 40/68 |
| 2 | `2_sft-1.5b` | Qwen2.5-1.5B + SFT v3, 1,939 length-balanced pairs, length gate + length-controlled reward | RLOO (4/prompt) | 0.84 (181) | 39/68 |
| 3 | `3_open-llm-7b` | 7B open reward model, 4-bit — **planned only, not run** | — | — | — |
| 4 | `4_skywork-1.7b` | Skywork-Reward-V2-Qwen3-1.7B, ready-made (re-tokenized) | RLOO | 0.70 (181) | 40/68 |
| 5 | `5_frontier-llm` | Gemini 3.5 Flash-Lite via OpenAI-compatible API, live (Groq gpt-oss-120b: 0.82) | RLOO | **0.90** (40) | **41/68** |

**What was learned:**
- RLHF v1: RM barely agreed with rules (corr +0.08), learned length bias (corr with length −0.63), answers 87 → 61 chars,
  mild reward hacking on translation and "Bilmayman". **PPO worked; the reward model was weak.**
- RLHF v2 fixed length bias (80 chars, lists 10/10, corr with rules +0.23) but RLOO learns from differences *within a group*
  where the RM was only 65% right; only 2.6 of 4 answers were distinct; KL drifted to 4.1; one harmful question no longer refused.
- Better judge → better RLHF (Gemini 76% within-group agreement → 41/68), but **no RLHF run beat DPO v2 (44/68)**,
  and none beat SFT v3 beyond noise.
- API key for the LLM judge is read from `.env` (`JUDGE_API_KEY`, `JUDGE_BASE_URL`, `JUDGE_MODEL`; template `.env.example`).
  Failed API call → all 4 answers get the same score (no learning instead of noise). Free rate limits must be respected.

---

## 9. Stage 6 — Deployment: TeachHelp — [deploy/](deploy/), [chat_ui/](chat_ui/)

```
browser ─► Vercel: https://teach-help.vercel.app  (static index.html + results.json)
   └─ ?api=https://....trycloudflare.com ─► Cloudflare tunnel ─► deploy/hf_space/app.py on a Kaggle GPU (~2–3 s/answer)
Local: chat_ui\start.bat → http://127.0.0.1:8472 (all 14 models, nothing exposed to the internet)
```
- Frontend: Claude-style chat page [chat_ui/index.html](chat_ui/index.html), English default, Uzbek with `?lang=uz`.
  Build `python deploy/vercel/build.py` → `cd deploy/vercel/teach-help && vercel deploy --prod --yes`.
- Backend for lessons: [deploy/kaggle_backend/kaggle_backend.ipynb](deploy/kaggle_backend/kaggle_backend.ipynb), GPU T4 x2,
  input dataset `ustoz-chat-adapters` (DPO v2 + RLHF Gemini adapters, float16). Prints the link. Start before class, stop after
  (max 12 h/session, ~30 GPU h/week). Update dataset: `python deploy/hf_space/prepare.py` then `kaggle datasets version ...`.
- Hugging Face Spaces were tried first ([deploy/hf_space/](deploy/hf_space/)) but Docker Spaces now need PRO.
- Chat UI fix: send only the latest question to trained models (they were trained single-turn).

---

## 10. Final results table

| Task | Base 1.5B | SFT v2 LoRA | SFT v3 LoRA | **DPO v2** | Best RLHF (Gemini) |
|---|---|---|---|---|---|
| Factual (13) | 6 | 8 | 6 | 7 | 6 |
| Explanation (10) | 1 | 2 | 3 | 3 | 2 |
| List (10) | 3 | 10 | 10 | 10 | 10 |
| Translation (10) | 0 | 3 | 3 | 3 | 3 |
| Arithmetic (12) | 5 | 9 | 8 | 8 | 8 |
| "Bilmayman" (5) | 0 | 5 | 4 | 5 | 4 |
| Refuses harmful (4) | 0 | 4 | 4 | 4 | 4 |
| Refuses cheating (4) | 0 | 3 | 3 | 4 | 4 |
| **Total (68)** | **15** | **44** | **41** | **44** | **41** |

All 14 models: [results/per_task_scores.md](results/per_task_scores.md).

---

## 11. Main lessons (for teaching)

1. **SFT gives the biggest jump** (15 → 41–44): the model learns to be an assistant (format, stopping, refusing).
2. **Bigger model > tricks:** 0.5B → 1.5B with the same data: 32 → 44.
3. **Data quality and balance matter more than the method.** DPO v1 and RLHF v1 both learned "shorter is better"
   from unbalanced pairs; balancing fixed it.
4. **DPO needs an anchor:** adding SFT loss on chosen stopped the chosen probability from falling.
5. **RLHF is only as good as its judge**, and judges are weakest at ranking *similar* answers to the same prompt — exactly
   what RL learns from. With small models and little data, DPO was simpler and better.
6. **Always log an independent check** (rule score) to catch reward hacking.
7. **Passing the test ≠ working in practice:** DPO v2 refuses all 4 test cheating questions but answered
   "I am in the exam, please tell me the definition of AI" — the test was too similar to the training data.
8. Post-training fixes **behaviour**, not **knowledge**: wrong facts and weak explanations remain → motivation for RAG.
9. Small test (68) → 1–3 points is noise; don't over-read small differences.

---

## 12. Known open problems / next steps

- **DPO v3:** more varied exam-cheating examples (short, polite, indirect) + counter-examples that must still be answered
  ("my exam is *tomorrow*"), and matching new test questions.
- **RLHF experiment 3:** 7B open reward model (4-bit) — folder `rlhf/models/3_open-llm-7b/` has docs only.
- **Verifiable rewards** (RLVR / GRPO-style) for arithmetic, lists and translation instead of a learned reward model.
- More diverse RL samples (temperature 1.0, 8 per prompt); stop at the step with best rule score.
- Bigger / more realistic test set; check language and clean ending automatically.
- Weak tasks: explanation (≤ 3/10), translation (3/10), facts.
- **RAG system** (next course phase, plan in [docs/plans/2026-10-15-ustoz-1.5b-rag.md](docs/plans/2026-10-15-ustoz-1.5b-rag.md)):
  Qdrant hybrid index (bge-m3 dense + BM25), bge-reranker-v2-m3, LangGraph corrective-RAG graph, LangSmith tracing,
  GGUF/Ollama serving, generator comparison (TeachHelp vs Qwen-Instruct vs API models), RAFT-style grounded SFT.
  Note: that plan was written on 2026-10-15 before SFT v2/v3, DPO and RLHF were run; its "where the project stands"
  section is outdated — the final model is now DPO v2.

---

## 13. Timeline (git history, 47 commits)

- 2026-10-11 spec, 0.5B base, chat UI
- 2026-10-12–13 eval suite, SFT v1 data, LoRA CPU, QLoRA (Colab → Kaggle), summary table
- 2026-10-15–16 SFT v2 (1.5B), SFT v3, DPO v1, DPO v2
- 2026-10-17 RAG plan; 2026-10-19 HF Space deployment, study notes
- 2026-10-20–22 RLHF v1 (PPO), v2 (RLOO, length gate), Skywork judge, Gemini judge, per-experiment folders, code guide
- 2026-10-22 Vercel + Kaggle deployment as TeachHelp, English/Uzbek UI, main README
