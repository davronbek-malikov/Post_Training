# Classic RLHF on top of SFT v3 LoRA: reward model + PPO (Qwen2.5-1.5B, Kaggle GPU)

DPO v2 reached the preference goal without RL. This folder does the same job the classic way
(InstructGPT, 2022), so the two methods can be compared on the **same preference pairs**
(`dpo/data_v2`) and the **same 68-question test**.

```
SFT v3 (done)  →  2. Reward model  →  3. PPO (online loop)
```

| Step | Script | Where | Output |
|---|---|---|---|
| 1. SFT | `sft_v2/train.py` (SFT v3 data) | done | `chat_ui/adapters/sft-lora-v3/` |
| 2. Reward model | `train_reward.py` | Kaggle GPU | `rlhf/models/1_base-0.5b/` (not in git) |
| 3. PPO | `train_ppo.py` | Kaggle GPU | `chat_ui/adapters/sft-lora-rlhf/`, `rlhf/logs/` |
| 4. Evaluation | `eval/run_eval.py --base Qwen2.5-1.5B --model sft-lora-rlhf` | Kaggle | `eval/results/sft-lora-rlhf.json` |

Everything from step 2 runs in `models/1_base-0.5b/rlhf_kaggle.ipynb`. Like DPO, it needs `sft_v3/sft_v3_adapters.zip`
uploaded to Kaggle as a dataset.

## The four models in memory

| Model | What it is here | Trained? | Device (T4 x2) |
|---|---|---|---|
| Policy | Qwen2.5-1.5B + SFT v3 LoRA, the LoRA keeps training | yes | GPU 0 |
| Reference | Qwen2.5-1.5B + SFT v3 LoRA, frozen copy, for the KL penalty | no | GPU 1 |
| Reward model | Qwen2.5-0.5B + LoRA + scalar head (`train_reward.py`) | no (trained in step 2) | GPU 1 |
| Value model (critic) | starts as a copy of the reward model, keeps training | yes | GPU 1 |

The reward model is smaller than the policy (InstructGPT also used a 6B reward model for a 175B policy).
Qwen2.5-0.5B and 1.5B share one tokenizer, so the reward model reads the policy's tokens directly.

## Reward model (`train_reward.py`)

- Input: prompt + answer. Output: one number, the score of the answer (taken at its last token).
- Loss (Bradley-Terry): `-log σ(r(chosen) − r(rejected))`. This is the DPO loss with
  `r = β · log(π / π_ref)` replaced by a separate model.
- Data: the 357 DPO v2 pairs (339 train, 18 validation). Rule-based labels from `dpo/label.py`, not human ones.
- Saved with the mean and standard deviation of its scores, so PPO can normalise them to mean 0, std 1.
- Length check: the script reports how often the reward model prefers the shorter answer and how strongly
  its score correlates with length. DPO v1 showed what a length bias does.

## PPO (`train_ppo.py`)

One step, repeated for every batch of 8 prompts (776 prompts from `dpo/data/prompts.jsonl`):

1. **Rollout**: the policy writes one answer per prompt (sampling, temperature 0.7).
2. **Score**: the reward model scores each answer; an answer that did not stop gets −1.
3. **KL penalty** per token: `−β · (log π − log π_ref)`, β = 0.05; the reward model score is added at the last token.
4. **Advantages** with GAE (γ = 1, λ = 0.95) from the value model's predictions.
5. **Update**: 2 PPO epochs over the batch. Policy: clipped ratio loss (ε = 0.2). Value: clipped MSE to the returns.

Sanity checks in the log: at step 1 the KL is ~0 (policy = reference, like DPO's 0.693 at the start).

**Reward hacking check.** Next to the reward model score, every step also logs the **rule score** from
`dpo/label.py`, which PPO never sees. If the reward model score rises while the rule score falls, the policy
is fooling the reward model. Every answer is saved to `rlhf/logs/ppo_rollouts.jsonl` for reading.

## Results: RLHF v1 (Kaggle, T4 x2)

**Reward model:** validation loss 0.719 → 0.508, accuracy 0.61 → 0.72 (only 18 validation pairs, so ±0.1 is noise).

**PPO:** 97 steps, 776 prompts, 1 epoch. Averages over each block of ~20 steps:

| Steps | RM score (raw) | Rule score | KL | Length (tokens) |
|---|---|---|---|---|
| 1–20 | 25.93 | 0.62 | 0.16 | 31 |
| 21–40 | 25.33 | 0.60 | 0.58 | 33 |
| 41–60 | 26.48 | 0.57 | 1.02 | 30 |
| 61–80 | 26.10 | 0.60 | 1.32 | 29 |
| 81–97 | 26.37 | 0.57 | 1.92 | 28 |

The KL check passed (0.000 at step 1), and every answer stopped on its own (100%).

**68-question test:**

| | SFT v3 | DPO v2 | RLHF (PPO) |
|---|---|---|---|
| Test score | 41/68 | **44/68** | 40/68 |
| Refuses harmful | 4/4 | 4/4 | 4/4 |
| Refuses cheating | 3/4 | 4/4 | 4/4 |
| "Bilmayman" | 4/5 | 5/5 | 4/5 |
| "Kechirasiz" inside normal answers | 1 | 0 | 1 |
| Average answer length (characters) | 87 | 76 | 61 |

Compared with DPO v2, RLHF fixed nothing new and lost four answers. All four losses are shorter answers:
- the list `1. Accuracy, 2. Precision, 3. Recall.` is written on one line, so the list check fails;
- "Tokenizer nima?" gets a shorter, vaguer explanation;
- one translation is wrong;
- an unanswerable question gets a refusal instead of "Bilmayman".

### What went wrong: the reward model, not PPO

PPO did its job: it raised the reward model's score while keeping KL small. The reward model was a weak judge:

- **It barely agrees with the rules.** Over all 776 PPO answers, the correlation between the RM score and the
  rule score is **+0.08**, close to none. Maximising the RM score therefore did not improve the answers.
- **It learned "shorter is better".** The correlation between the RM score and answer length is **−0.63**.
  This is the same length bias as DPO v1, now inside the reward model, so answers became 30% shorter (87 → 61 characters).
- **Mild reward hacking on some tasks.** Between the first and second half of training the RM score rose
  while the rule score fell: translation (RM 25.97 → 27.33, rule 0.75 → 0.50) and "Bilmayman" (RM 27.66 → 28.70, rule 0.50 → 0.38).

Why the reward model is weak: 339 training pairs, a 0.5B base model that has never been fine-tuned on these tasks,
and labels from rules, not humans. DPO v2 used the same 339 pairs but did not have to compress them into one
score first. This is the main lesson: **in classic RLHF the policy can only become as good as its reward model**,
and a reward model needs much more data than 339 pairs.

## TODO

- [x] Folder, plan, TODO list
- [x] `scalar_model.py`: transformer + scalar head, shared by the reward and value models
- [x] `train_reward.py`: Bradley-Terry training, validation accuracy, length-bias report, score normalisation
- [x] `train_ppo.py`: rollout → reward → KL → GAE → clipped PPO update, rule-score logging
- [x] CPU smoke tests with Qwen2.5-0.5B (`--max-steps`)
- [x] `models/1_base-0.5b/rlhf_kaggle.ipynb`
- [x] Run on Kaggle: reward model (validation accuracy 0.72)
- [x] Run on Kaggle: PPO (97 steps, KL 0 → 1.9)
- [x] Evaluate on the 68-question test (40/68), add results here and to the chat UI description
- [x] Compare SFT v3 vs DPO v2 vs RLHF
- [x] RLHF v2 code: `build_rm_data.py`, length gate, 1.5B reward model, `--algo rloo`, `models/2_sft-1.5b/rlhf_v2_kaggle.ipynb`
- [ ] Run `models/2_sft-1.5b/rlhf_v2_kaggle.ipynb` on Kaggle, unzip `rlhf_v2_results.zip`, run `python eval/summarize.py`
- [ ] If RLHF v2 beats DPO v2, deploy it: `python deploy/hf_space/prepare.py` + `hf upload` (see `deploy/README.md`)

## RLHF v2 (`models/2_sft-1.5b/rlhf_v2_kaggle.ipynb`)

Each change fixes one of the v1 causes listed above:

| v1 problem | v2 fix | Where |
|---|---|---|
| 339 pairs, one per prompt | every pair of the 8 answers + the reference whose rule scores differ by ≥ 0.5: **1 939 pairs** | `build_rm_data.py` |
| RM learned "shorter is better" | per task, as many pairs with a shorter chosen as with a longer one (50/50); for refusals the longer SFT refusal is also a chosen answer | `build_rm_data.py` |
| 18 validation pairs, split by pair | 181 pairs from 78 held-out **prompts** | `build_rm_data.py` |
| RM = base 0.5B, has never seen the tasks | RM = Qwen2.5-1.5B **+ SFT v3**, loaded in float16 during RL | `train_reward.py --sft-adapter sft-lora-v3` |
| PPO started even with a biased RM | **length gate**: the RM must not pick the shorter answer >15 points more often than the data, and within each pair the score difference must not track the length difference (corr ≥ −0.3; a correlation across different prompts is meaningless, since each task has its own average score); otherwise PPO refuses to start | `train_reward.py`, `train_ppo.py` |
| RM still gave shorter answers a bigger margin (Kaggle run: val acc 0.84, picks shorter 53%, but pair_corr −0.35) | **length-controlled reward**: score − `length_coef` × characters, the slope measured on train pairs, the gate checked on val | `train_reward.py`, `train_ppo.py` |
| noisy value model (vf_loss 1–7): different prompts have different average RM scores | **RLOO**: 4 answers per prompt, baseline = mean reward of the other 3; no value model | `train_ppo.py --algo rloo` |
| KL 0.05 let the policy drift to 1.9 | KL 0.1 | notebook |

```
python rlhf/build_rm_data.py
python rlhf/train_reward.py --base Qwen2.5-1.5B --sft-adapter sft-lora-v3 --data rlhf/data_v2 --out 2_sft-1.5b
python rlhf/train_ppo.py --algo rloo --group-size 4 --reward 2_sft-1.5b --reward-fp16 --kl-coef 0.1 \
    --name sft-lora-rlhf-v2 --rollouts rlhf_v2_rollouts.jsonl
```

What to watch: RM score **and** rule score rising together, answer length staying near 30 tokens, and on the test,
List and "Bilmayman" back to 10/10 and 5/5. If rule score falls while the RM score rises, the reward model is still
being fooled. The next step would then be human (or stronger-model) labels, not more RL.

## CPU smoke test (laptop)

```
python rlhf/train_reward.py --base Qwen2.5-0.5B --max-steps 2
python rlhf/train_ppo.py --base Qwen2.5-0.5B --sft-adapter sft-lora --max-steps 1 --batch-size 2 --mini-batch 1 --max-new-tokens 16
```
Neither command saves anything to `chat_ui/adapters/`. The reward model test saves to `rlhf/models/1_base-0.5b/`,
which is needed for the PPO test.

## Ready-made judges instead of our own reward model

Two more rows of the reward-model table, run with the same RLOO as v2 (`train_ppo.py --algo rloo`):

| # | Judge | Notebook | Option | Output |
|---|---|---|---|---|
| 4 | Ready-made reward model `Skywork/Skywork-Reward-V2-Qwen3-1.7B` (no training; Qwen3 tokenizer, so answers are re-tokenized) | `models/4_skywork-1.7b/rlhf_skywork_kaggle.ipynb` | `--judge hf --judge-model ...` | `sft-lora-rlhf-skywork` |
| 5 | A large LLM through an OpenAI-compatible API (default: Groq `openai/gpt-oss-120b`), live during RL; one call scores the 4 answers of one prompt in shuffled order | `models/5_frontier-llm/rlhf_llm_judge_kaggle.ipynb` | `--judge api` + `JUDGE_API_KEY` | `sft-lora-rlhf-llm` |

Both judges have their own score scale, so scores are standardised per step. A failed API call gives all 4 answers
the same score, so RLOO learns nothing from that prompt instead of learning from noise.
Before RL, `python rlhf/eval_judge.py --judge hf|api` scores the 181 validation pairs, for comparison with our v2 reward model (0.84).
The code is in `rlhf/judges.py`.
