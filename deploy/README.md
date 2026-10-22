# Bepul deploy: Hugging Face Space (backend) + Vercel (frontend)

```
brauzer ──► Vercel (index.html, results.json, config.js) ── statik, bepul
   │
   └─ fetch(API_BASE + "/api/generate") ──► Hugging Face Space (Docker, CPU basic: 2 vCPU, 16 GB) ── bepul
```

Faqat eng yaxshi modellar: bitta Qwen2.5-1.5B base va 2 ta adapter — DPO v2 (yakuniy model, 44/68) va eng yaxshi RLHF (Gemini hakam, 41/68).
Adapterlar float16'da (~140 MB har biri). Bepul Space 48 soat ishlatilmasa uxlaydi; birinchi so'rovda 1-2 daqiqada
uyg'onadi (sahifa o'zi qayta urinadi). Javob ~20 soniya (CPU).

## 1. Backend: Hugging Face Space

```
hf auth login                                   # bir marta, "write" token bilan
python deploy/hf_space/prepare.py               # adapterlar -> deploy/hf_space/adapters/ (float16)
hf repos create <user>/ustoz --repo-type space --space-sdk docker   # bir marta
hf upload <user>/ustoz deploy/hf_space . --repo-type space
```

Build ~10 daqiqa (torch + base model yuklab olinadi). Tekshirish: `https://<user>-ustoz.hf.space/api/models`.
Yangi model (masalan, RLHF v2) qo'shilganda: `prepare.py` va `hf upload` ni qayta ishga tushiring.

## 2. Frontend: Vercel

```
python deploy/vercel/build.py --api https://<user>-ustoz.hf.space
vercel deploy deploy/vercel/dist --prod
```

`chat_ui/index.html` o'zgarishsiz ishlatiladi: faqat `config.js` boshqa (lokal serverda `API_BASE = ''`).
