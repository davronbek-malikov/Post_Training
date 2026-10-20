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
| 2. Reward model | `train_reward.py` | Kaggle GPU | `rlhf/models/reward/` (not in git) |
| 3. PPO | `train_ppo.py` | Kaggle GPU | `chat_ui/adapters/sft-lora-rlhf/`, `rlhf/logs/` |
| 4. Evaluation | `eval/run_eval.py --base Qwen2.5-1.5B --model sft-lora-rlhf` | Kaggle | `eval/results/sft-lora-rlhf.json` |

Everything from step 2 runs in `rlhf_kaggle.ipynb`. Like DPO, it needs `sft_v3/sft_v3_adapters.zip`
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

## TODO

- [x] Folder, plan, TODO list
- [x] `scalar_model.py`: transformer + scalar head, shared by the reward and value models
- [x] `train_reward.py`: Bradley-Terry training, validation accuracy, length-bias report, score normalisation
- [x] `train_ppo.py`: rollout → reward → KL → GAE → clipped PPO update, rule-score logging
- [x] CPU smoke tests with Qwen2.5-0.5B (`--max-steps`)
- [x] `rlhf_kaggle.ipynb`
- [ ] Run on Kaggle: reward model (check validation accuracy and length bias before PPO)
- [ ] Run on Kaggle: PPO (watch KL, reward model score vs rule score)
- [ ] Evaluate on the 68-question test, add results here and to the chat UI description
- [ ] Compare SFT v3 vs DPO v2 vs RLHF: score, refusals, "Bilmayman", answer length, compute time
- [ ] Optional: more reward-model pairs (all 8 answers per prompt, not one pair), larger reward model (1.5B)

## CPU smoke test (laptop)

```
python rlhf/train_reward.py --base Qwen2.5-0.5B --max-steps 2
python rlhf/train_ppo.py --base Qwen2.5-0.5B --sft-adapter sft-lora --max-steps 1 --batch-size 2 --mini-batch 1 --max-new-tokens 16
```
Neither command saves anything to `chat_ui/adapters/`. The reward model test saves to `rlhf/models/reward/`,
which is needed for the PPO test.
