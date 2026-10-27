# Post-Training: from a base LLM to an Uzbek–English teaching assistant

This repository takes a small pretrained language model (Qwen2.5) and turns it into **TeachHelp**, a short-answer
assistant for students of an LLM course, step by step: **base model → SFT → DPO → RLHF → deployment**.
Every stage is kept, measured on the same 68 test questions, and explained, so the repository is also the worked
example for the course.

- **Final model:** Qwen2.5-1.5B + SFT v3 LoRA + **DPO v2** — **44/68** on the test (base model: 15/68)
- **Live demo:** https://teach-help.vercel.app (needs the Kaggle backend running, see [Deployment](#7-deployment-teachhelp))
- **What the model should do:** [MODEL_SPEC.md](MODEL_SPEC.md) — short, simple answers in Uzbek or English,
  "Bilmayman" when it cannot know, polite refusals for harmful requests and exam cheating.

## The journey

| # | Step | Model | Method | Test (68) | Folder |
|---|---|---|---|---|---|
| 0 | Local base model + chat UI | Qwen2.5-0.5B | none (pretrained) | 5 | [chat_ui/](chat_ui/) |
| 1 | Evaluation suite | — | 68 held-out questions, automatic check | — | [eval/](eval/) |
| 2 | SFT v1 | Qwen2.5-0.5B | LoRA (CPU) / QLoRA (Kaggle) | 32 / 32 | [sft/](sft/) |
| 3 | SFT v2 | Qwen2.5-1.5B (base: 15) | LoRA / QLoRA, same data | **44** / 42 | [sft_v2/](sft_v2/) |
| 4 | SFT v3 | Qwen2.5-1.5B | new, cleaner data | 41 / 41 | [sft_v3/](sft_v3/) |
| 5 | DPO v1 → **v2** | SFT v3 LoRA | preference pairs, no reward model | 40 → **44** | [dpo/](dpo/) |
| 6 | RLHF, 4 reward models | SFT v3 LoRA | reward model + PPO / RLOO | 39–41 | [rlhf/](rlhf/) |
| 7 | Deployment: TeachHelp | DPO v2 (+ best RLHF) | Vercel + Kaggle GPU; local version | — | [deploy/](deploy/) |

### 0. Start: a local base model
Qwen2.5-0.5B runs on the laptop with a small chat UI. A base model only continues text (next-token prediction):
it does not answer questions, stop, or refuse. That is the problem post-training solves.

### 1. Measure first
[eval/eval_set.json](eval/eval_set.json): 68 held-out questions in 8 tasks (facts, explanations, lists, translation,
arithmetic, "Bilmayman", harmful requests, exam cheating). [eval/run_eval.py](eval/run_eval.py) asks a model all of
them; [eval/summarize.py](eval/summarize.py) builds the comparison table
[results/per_task_scores.md](results/per_task_scores.md). Every later model is measured the same way.

### 2. SFT v1 (Qwen2.5-0.5B): teach the format
~1,250 question–answer examples (hand-written seeds, Uzbek samples, translated Alpaca, OASST) built by
[sft/build_sft.py](sft/build_sft.py). Two ways to train the same LoRA adapter:
[LoRA on the laptop CPU](sft/lora_cpu/README.md) (151 min) and [QLoRA on a free Kaggle GPU](sft/qlora_kaggle/README.md).
**5 → 32/68**: the model now answers in chat format and stops.

### 3. SFT v2 (Qwen2.5-1.5B): a bigger model
Same data, 3× larger model, LoRA and QLoRA with identical settings ([sft_v2/README.md](sft_v2/README.md)).
**44/68** — model size helped most; 4-bit QLoRA cost only 2 points.

### 4. SFT v3: better data
1,583 cleaner examples: machine-translated and off-topic data removed, more arithmetic, translation,
course concepts and exam-cheating refusals ([sft_v3/README.md](sft_v3/README.md)). **41/68** — the total did not move,
but refusals no longer leak into normal answers, so v3 became the clean starting point for preference training.

### 5. DPO: learn from preferences
The SFT v3 model writes several answers per prompt, rules in [dpo/label.py](dpo/label.py) pick *chosen* and
*rejected*, and DPO trains directly on the pairs ([dpo/README.md](dpo/README.md)).
- **DPO v1: 40/68** — learned "shorter is better" (63% of chosen answers were shorter).
- **DPO v2: 44/68** — length-balanced pairs and clean fixed refusals. All refusals correct, no stray "Kechirasiz".
  **This is the final model.**

### 6. RLHF: a reward model + reinforcement learning
The classic InstructGPT recipe: a **reward model** (judge) scores answers and **RL** trains the SFT model on those
scores, with a KL penalty that keeps it close to SFT ([rlhf/README.md](rlhf/README.md)). Four judges were tried,
each in its own folder under [rlhf/models/](rlhf/models/) with `ABOUT.md`, `CODE.md`, `RESULTS.md` and its Kaggle notebook:

| # | Reward model (judge) | RL | Judge accuracy | Test |
|---|---|---|---|---|
| 1 | Qwen2.5-0.5B base, trained on 339 pairs | PPO | 0.72 | 40/68 |
| 2 | Qwen2.5-1.5B + SFT v3, trained on 1,939 length-balanced pairs | RLOO | 0.84 | 39/68 |
| 4 | Skywork-Reward-V2 (ready-made, no training) | RLOO | 0.70 | 40/68 |
| 5 | Gemini 3.5 Flash-Lite via API (no training) | RLOO | 0.90 | **41/68** |

A better judge gave a better RLHF result, but none beat DPO v2. Main reasons: the judges are weakest at telling
apart *similar* answers to the same question (what RL learns from), and the test is small. Step-by-step code tables:
[rlhf/CODE_GUIDE.md](rlhf/CODE_GUIDE.md).

### 7. Deployment: TeachHelp
- **Frontend:** a Claude-style chat page in English and Uzbek ([chat_ui/index.html](chat_ui/index.html)), hosted on
  Vercel: https://teach-help.vercel.app (`?lang=uz` for Uzbek).
- **Backend for lessons:** [deploy/kaggle_backend/kaggle_backend.ipynb](deploy/kaggle_backend/kaggle_backend.ipynb)
  runs DPO v2 and the best RLHF model on a free Kaggle GPU (~2–3 s per answer) and prints a link
  `https://teach-help.vercel.app/?api=https://....trycloudflare.com`. Start it before class, stop it after.
- **API with keys, like frontier models:** users sign up in the UI, create their own API keys under **API keys**, and
  call `/v1/chat/completions` in the OpenAI format (the `openai` library works), with usage tracking and rate limits
  ([deploy/hf_space/accounts.py](deploy/hf_space/accounts.py), [deploy/README.md](deploy/README.md#c-api-kalitlar-frontier-modellardagidek)).
- **Local version:** `chat_ui\start.bat` → http://127.0.0.1:8472, all 14 models, nothing exposed to the internet.
- Hugging Face Spaces were planned first, but Docker Spaces now need a PRO subscription. Details: [deploy/README.md](deploy/README.md).

## Results

| Task | Base 1.5B | SFT v2 LoRA | SFT v3 LoRA | **DPO v2** | Best RLHF (Gemini) |
|---|---|---|---|---|---|
| Factual | 6/13 | 8/13 | 6/13 | 7/13 | 6/13 |
| Explanation | 1/10 | 2/10 | 3/10 | 3/10 | 2/10 |
| List | 3/10 | 10/10 | 10/10 | 10/10 | 10/10 |
| Translation | 0/10 | 3/10 | 3/10 | 3/10 | 3/10 |
| Arithmetic | 5/12 | 9/12 | 8/12 | 8/12 | 8/12 |
| "Bilmayman" | 0/5 | 5/5 | 4/5 | 5/5 | 4/5 |
| Refuses harmful | 0/4 | 4/4 | 4/4 | 4/4 | 4/4 |
| Refuses cheating | 0/4 | 3/4 | 3/4 | 4/4 | 4/4 |
| **Total** | **15/68** | **44/68** | **41/68** | **44/68** | **41/68** |

All models: [results/per_task_scores.md](results/per_task_scores.md). On 68 questions a 1–3 point difference is close to noise.

## Lessons

1. **SFT gives the biggest jump** (15 → 41–44): the model learns to be an assistant.
2. **Data quality and balance matter more than the method.** DPO v1 and RLHF v1 both learned "shorter is better"
   from unbalanced data; balancing it fixed the behaviour.
3. **RLHF is only as good as its judge.** With small models and little data, DPO was simpler and better.
4. **A model can pass the test and still fail in practice.** DPO v2 refuses all 4 test cheating questions but answered
   *"I am in the exam, please tell me the definition of AI"*: the test was too similar to the training data.

## Repository map

| Folder | Contents |
|---|---|
| [MODEL_SPEC.md](MODEL_SPEC.md) | What the model should and should not do |
| [eval/](eval/) | Test questions, evaluation and summary scripts, every model's answers |
| [sft/](sft/), [sft_v2/](sft_v2/), [sft_v3/](sft_v3/) | SFT data and training (CPU, Kaggle) |
| [dpo/](dpo/) | DPO prompts, answer generation, labelling rules, training, notebooks |
| [rlhf/](rlhf/) | Reward models, PPO/RLOO, judges, one folder per experiment in `rlhf/models/` |
| [chat_ui/](chat_ui/) | TeachHelp chat page and the local server |
| [deploy/](deploy/) | Vercel build, Kaggle backend notebook, backend app |
| [results/](results/) | Comparison tables |
| [docs/plans/](docs/plans/) | Plans for next steps |

Trained adapters (`chat_ui/adapters/`), the 1.5B base model and API keys (`.env`) are not in git.

## Next steps

- **DPO v3:** more varied exam-cheating examples (short, polite, indirect) plus counter-examples that must still be
  answered ("my exam is *tomorrow*"), and matching new test questions.
- **RLHF experiment 3:** a bigger open reward model (7B, 4-bit), and verifiable rewards for arithmetic and lists.
