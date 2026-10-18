---
title: Ustoz DPO v2
emoji: 🎓
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

API backend for the Post_Training chat UI: Qwen2.5-1.5B + SFT v3 LoRA + DPO v2, served on CPU.

- `GET /api/models`: the one available model
- `POST /api/generate`: `{"model": "sft-lora-dpo-v2", "messages": [{"role": "user", "content": "..."}]}` → streamed text

Source: https://github.com/davronbek-malikov/Post_Training (`deploy/hf_space/`).
