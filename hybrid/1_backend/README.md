# 1. Backend: gateway

Foydalanuvchi va model o'rtasidagi "darvozabon". Frontier API'lar (OpenAI, Anthropic) ham shunday tuzilgan:
oldida gateway (kalit, limit, hisob), orqasida model serveri.

| Fayl | Nima qiladi |
|---|---|
| [accounts.py](accounts.py) | SQLite baza: foydalanuvchilar (parol scrypt xeshi), sessiyalar, `sk-th-...` kalitlar (sha256 xeshi), kalitning modeli va muddati, rate limit (minutiga) va kunlik limit, sarf va narx (`PRICES`, $ / 1M token), admin endpointlar (`/admin/*`) |
| [gateway.py](gateway.py) | FastAPI ilova: `/v1/chat/completions` (OpenAI formati, stream), `/v1/models`, chat sahifa uchun `/api/*`. So'rovni `model` nomiga qarab `:8080` (bizning model) yoki `:8081` (Qwen3.5-4B) ga yuboradi |

Bitta so'rovning yo'li:

```
POST /v1/chat/completions  (Authorization: Bearer sk-th-...)
  1. accounts.check_api_key   kalit bormi, bekor qilinmaganmi, muddati o'tmaganmi, limitlar
  2. accounts.check_model     kalit shu modelga ruxsat beradimi
  3. gateway.engine_open      so'rovni llama-server'ga yuboradi
  4. gateway.engine_stream    javobni bo'laklab (SSE) foydalanuvchiga uzatadi
  5. accounts.record_usage    tokenlar va narx bazaga yoziladi
```

Alohida ishga tushirish (sinov uchun): `cd hybrid/1_backend && uvicorn gateway:app --port 7860`.
Odatda buni [5_runner/run.py](../5_runner/run.py) qiladi. `accounts.py` ni Kaggle backend
([deploy/hf_space/app.py](../../deploy/hf_space/app.py)) ham ishlatadi.
