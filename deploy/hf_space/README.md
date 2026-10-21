---
title: Ustoz Post-Training
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

API backend for the Post_Training chat UI: Qwen2.5-1.5B + SFT v3 LoRA, followed by DPO v2 or RLHF, served on a free CPU Space.
Only the best models are deployed: one base model, with each adapter stored in float16 (~140 MB each).

- `GET /api/models`: the deployed models
- `POST /api/generate`: `{"model": "sft-lora-dpo-v2", "messages": [{"role": "user", "content": "..."}]}` → streamed text

Source: https://github.com/davronbek-malikov/Post_Training (`deploy/hf_space/`).
