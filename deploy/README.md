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
python hybrid/3_ui/build.py                        # -> hybrid/3_ui/teach-help/
cd hybrid/3_ui/teach-help && vercel deploy --prod --yes
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
3. Ixtiyoriy: **Add-ons -> Secrets -> `HF_TOKEN`** (write) — hisoblar va API kalitlar bazasi sessiyalar orasida saqlanadi
4. Editor'da **Run All** -> har bosqich o'zini tekshiradi (fayllar, base model mosligi, model javoblari, API testi),
   8-katakda havolalar chiqadi: UI havolasi va API base URL. Tugatganda **⏻ Stop session**.

## C. API kalitlar (frontier modellardagidek)

```
foydalanuvchi ──UI──► Ro'yxatdan o'tish / Kirish ──► "API kalitlar" menyusi ──► sk-th-... kalit (faqat bir marta ko'rinadi)
dastur ──Authorization: Bearer sk-th-...──► /v1/chat/completions (OpenAI formati) ──► javob + token sarfi
```

| Fayl | Vazifasi |
|---|---|
| `hybrid/1_backend/accounts.py` | SQLite: foydalanuvchilar (parol scrypt xeshi), sessiyalar, kalitlar (sha256 xeshi), sarf, rate limit |
| `deploy/hf_space/app.py` | model + `/api/*` (UI, kalitsiz) + `/v1/models`, `/v1/chat/completions` (kalit bilan, stream ham) |
| `deploy/hf_space/test_api.py` | ro'yxatdan o'tish -> kalit -> so'rovlar -> bekor qilish: to'liq sinov |
| `chat_ui/index.html` | chap pastda **API kalitlar** va **Kirish**: kalit yaratish, ro'yxat, bekor qilish, sarf, kod namunalari |

```python
from openai import OpenAI
client = OpenAI(base_url="https://....trycloudflare.com/v1", api_key="sk-th-...")
r = client.chat.completions.create(model="sft-lora-dpo-v2", messages=[{"role": "user", "content": "Python nima?"}])
print(r.choices[0].message.content, r.usage)
```

Xatolar OpenAI formatida: 401 (kalit yo'q / noto'g'ri / bekor qilingan), 429 (minutiga `RATE_LIMIT` so'rovdan ko'p),
404 (model yo'q). Hujjat: `<server>/docs`. Lokal server (`chat_ui/server.py`) faqat UI uchun: unda hisoblar yo'q.
Taklif kodi (`SIGNUP_CODE`) berilsa, faqat kodni bilganlar ro'yxatdan o'tadi — havola ochiq internetda.

Cheklovlar: bir vaqtda bitta generatsiya (boshqalar navbatda); tunnel manzili har sessiyada yangi, shuning uchun
`base_url` ham o'zgaradi. Doimiy xizmat uchun keyingi qadamlar: vLLM (continuous batching), doimiy server va domen.

## D. Lokal server: o'z kompyuteringiz, Cloudflare tunnel orqali (hybrid)

Kaggle — dars uchun (GPU, vaqtincha). Doimiy server uchun alohida papka: [hybrid/](../hybrid/README.md) — model GGUF
formatida llama.cpp bilan CPU'da, oldida `hybrid/1_backend/gateway.py` (hisoblar, kalitlar, muddat, narxlar, limitlar,
**admin panel**), ikkinchi model sifatida tayyor Qwen3.5-4B, internetga Cloudflare tunnel, UI Vercel'da.
Papka 5 qismga bo'lingan: backend, engine, UI, tunnel, ularni bog'lovchi `run.py`.

Bitta sessiya ko'pi bilan 12 soat, haftada ~30 GPU soat. Kaggle — doimiy hosting emas: dars va namoyish uchun.
Datasetni yangilash: `python deploy/hf_space/prepare.py`, keyin
`kaggle datasets version -p deploy/hf_space/adapters --dir-mode zip -m "..."`.
