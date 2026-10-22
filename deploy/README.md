# Deploy: Vercel (frontend) + o'zimizning backend (lokal kompyuter yoki Kaggle GPU)

```
brauzer ──► Vercel: https://teach-help.vercel.app   (index.html, results.json — statik, bepul, doim ochiq)
   │
   └─ ?api=https://....trycloudflare.com ──► Cloudflare tunnel (bepul) ──► backend: deploy/hf_space/app.py
                                                                            ├─ A. lokal kompyuter (CPU, ~20 s)
                                                                            └─ B. Kaggle notebook (GPU, ~2-3 s)
```

Backendda faqat eng yaxshi modellar: Qwen2.5-1.5B + **DPO v2** (yakuniy model, 44/68) + **RLHF (Gemini hakam, 41/68)**.
Hugging Face Space endi ishlatilmaydi: Docker Space'lar bepul tarifda PRO obuna talab qiladi.

## Frontend: Vercel

```
python deploy/vercel/build.py                      # -> deploy/vercel/teach-help/
cd deploy/vercel/teach-help && vercel deploy --prod --yes
```

Sahifa backend manzilini havoladan oladi: `https://teach-help.vercel.app/?api=<backend>` va brauzerda eslab
qoladi. Havolasiz ochilsa, "Server ulanmagan" deb yozadi. `chat_ui/index.html` o'zgarganda qayta deploy qiling.

## A. Lokal backend ("enterprise" versiya: model sizning kompyuteringizda)

```
python deploy/local/start_backend.py        # yoki deploy/local/start_backend.bat ni ikki marta bosing
```

1-2 daqiqadan keyin tayyor havolani chiqaradi. Oyna ochiq turganda ishlaydi, to'xtatish — Ctrl+C. Kerak:
`Qwen2.5-1.5B/` va `chat_ui/adapters/sft-lora-dpo-v2`, `chat_ui/adapters/sft-lora-rlhf-llm`. Tunnel dasturi
(cloudflared) birinchi ishga tushishda o'zi yuklab olinadi. Kompyuter ~6 GB RAM ishlatadi.

## B. Kaggle backend (dars uchun: GPU, tez javob)

1. Kaggle -> **New Notebook** -> **File -> Import Notebook** -> `deploy/kaggle_backend/kaggle_backend.ipynb`
2. **GPU T4 x2**, **Internet On**, **Add Input** -> `ustoz-chat-adapters` (shaxsiy dataset: ikkala adapter, float16)
3. Editor'da **Run All** -> 5-katakda havola chiqadi. Tugatganda **⏻ Stop session**.

Bitta sessiya ko'pi bilan 12 soat, haftada ~30 GPU soat. Kaggle — doimiy hosting emas: dars va namoyish uchun.
Datasetni yangilash: `python deploy/hf_space/prepare.py`, keyin
`kaggle datasets version -p deploy/hf_space/adapters --dir-mode zip -m "..."`.
