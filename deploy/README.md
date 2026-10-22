# Deploy: TeachHelp — Vercel (frontend) + Kaggle GPU (backend); lokal versiya alohida

```
brauzer ──► Vercel: https://teach-help.vercel.app   (index.html, results.json — statik, bepul, doim ochiq)
   │
   └─ ?api=https://....trycloudflare.com ──► Cloudflare tunnel (bepul) ──► backend: deploy/hf_space/app.py
                                                                            └─ Kaggle notebook (GPU, ~2-3 s)
Lokal versiya alohida: chat_ui/start.bat -> http://127.0.0.1:8472 (internetsiz, faqat sizning kompyuteringizda)
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

## A. Lokal versiya (faqat sizning kompyuteringizda)

```
chat_ui\start.bat          # yoki: cd chat_ui && python server.py   ->   http://127.0.0.1:8472
```

Tunnel yo'q, internetga hech narsa ochilmaydi: sahifa ham, model ham shu kompyuterda. Barcha modellar (base,
SFT, DPO, RLHF) ro'yxatda. Vercel'dagi sahifa lokal serverga ulanmaydi — u Kaggle backend uchun.

## B. Kaggle backend (dars uchun: GPU, tez javob)

1. Kaggle -> **New Notebook** -> **File -> Import Notebook** -> `deploy/kaggle_backend/kaggle_backend.ipynb`
2. **GPU T4 x2**, **Internet On**, **Add Input** -> `ustoz-chat-adapters` (shaxsiy dataset: ikkala adapter, float16)
3. Editor'da **Run All** -> 5-katakda havola chiqadi. Tugatganda **⏻ Stop session**.

Bitta sessiya ko'pi bilan 12 soat, haftada ~30 GPU soat. Kaggle — doimiy hosting emas: dars va namoyish uchun.
Datasetni yangilash: `python deploy/hf_space/prepare.py`, keyin
`kaggle datasets version -p deploy/hf_space/adapters --dir-mode zip -m "..."`.
