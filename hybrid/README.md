# Hybrid deployment: TeachHelp API (UI bulutda, backend o'z kompyuteringizda)

Kompyuteringiz server bo'ladi: model va baza sizda (on-premise), UI Vercel'da (cloud), ular o'rtasida Cloudflare
tunnel. Foydalanuvchilar istalgan joydan ro'yxatdan o'tib `sk-th-...` kalit yaratadi va API'ni ishlatadi, siz admin
panelda kim qancha ishlatganini ko'rasiz. Model qayta o'qitilmaydi: tayyor DPO v2 GGUF formatiga o'tkaziladi va
llama.cpp bilan CPU'da ishlaydi.

```
                                  ┌──────────────── Sizning PC (on-premise) ─────────────────────────┐
brauzer ──► Vercel (3. UI) ──►    │ 4. Cloudflare   1. gateway (:7860)        2. llama-server :8080  │ bizning model
dastur (sk-th-... kalit) ───────► │    tunnel  ──►  kalit, limit, narx,  ──►  2. llama-server :8081  │ Qwen3.5-4B
                                  │                 sarf, admin                                      │
                                  │ 5. run.py — hammasini yoqadi, kuzatadi, tunnel manzilini Vercel'ga yozadi │
                                  └──────────────────────────────────────────────────────────────────┘
```

## Papka tuzilishi

| # | Qism | Qayerda ishlaydi | O'zimiz yozgan kod | Tayyor dastur |
|---|---|---|---|---|
| 1 | **Backend: gateway** | Sizning PC | [1_backend/accounts.py](1_backend/accounts.py), [1_backend/gateway.py](1_backend/gateway.py) | FastAPI, uvicorn |
| 2 | **Backend: model (engine)** | Sizning PC | Yo'q, faqat sozlamalar ([5_runner/run.py](5_runner/run.py) ichida). Modelni tayyorlash: [2_engine/merge.py](2_engine/merge.py) | `llama-server.exe` |
| 3 | **UI** | Vercel | [chat_ui/index.html](../chat_ui/index.html), [3_ui/build.py](3_ui/build.py) | Vercel |
| 4 | **Tunnel** | Cloudflare | Yo'q, faqat ishga tushirish ([5_runner/run.py](5_runner/run.py) ichida) | `cloudflared.exe` |
| 5 | **Hammasini bog'lovchi** | Sizning PC | [5_runner/run.py](5_runner/run.py), [start.bat](5_runner/start.bat), [stop.bat](5_runner/stop.bat) | — |

Har bir papkada qisqa README bor: nima qiladi, qanday ishlaydi.

```
hybrid/
├── 1_backend/    accounts.py (hisoblar, kalitlar, limit, narx), gateway.py (API, modelga yo'naltirish)
├── 2_engine/     merge.py (LoRA -> to'liq model), eval_api.py (68 savolli baholash), README (llama-server)
├── 3_ui/         build.py (Vercel sahifasini yig'ish), teach-help/ (yig'ilgan sahifa, git'da yo'q)
├── 4_tunnel/     README (Cloudflare tunnel qanday ishlaydi)
├── 5_runner/     run.py, start.bat, stop.bat, install_autostart.bat, uninstall_autostart.bat
├── bin/          llama-server.exe, cloudflared.exe          (git'da yo'q)
├── data/         teachhelp.db, settings.env, loglar          (git'da yo'q)
└── models/       GGUF modellar                               (git'da yo'q)
```

**O'qish tartibi:** `1_backend/accounts.py` → `1_backend/gateway.py` → `5_runner/run.py` → `3_ui/build.py` →
`chat_ui/index.html` (faqat `fetch(...)` chaqiruvlari). 2 va 4 qismlarda kodni emas, tushunchani o'qing.

`index.html` bitta joyda, `chat_ui/` da qoladi: uni Kaggle darsi (1-qism) va natijalar sahifasi ham ishlatadi.
`build.py` uni Vercel uchun nusxalaydi. Test: [1_backend/test_api.py](1_backend/test_api.py) — 8 bosqichli API testi, istalgan server
manzili bilan ishlaydi.

## 1. Bir martalik tayyorgarlik

| Kerak | Qayerda | Qanday |
|---|---|---|
| GGUF model | `hybrid/models/teachhelp-1.5b-q8_0.gguf` | pastda, "Modelni tayyorlash" |
| llama.cpp (Windows, CPU) | `hybrid/bin/llama/llama-server.exe` | https://github.com/ggml-org/llama.cpp/releases -> `llama-<versiya>-bin-win-cpu-x64.zip` -> `bin/llama/` ga oching |
| cloudflared | `hybrid/bin/cloudflared.exe` | https://github.com/cloudflare/cloudflared/releases -> `cloudflared-windows-amd64.exe` |
| Python kutubxonalari | | `pip install fastapi uvicorn httpx` |

### Modelni tayyorlash (bir marta)

```
python hybrid/2_engine/merge.py                                 # LoRA + base -> hybrid/models/teachhelp-1.5b/ (~3 GB)
git clone --depth 1 https://github.com/ggml-org/llama.cpp       # GGUF konvertori
pip install gguf
python llama.cpp/convert_hf_to_gguf.py hybrid/models/teachhelp-1.5b --outtype q8_0 \
       --outfile hybrid/models/teachhelp-1.5b-q8_0.gguf         # ~1.6 GB
```
8-bit (q8_0) tanlangan: 1.5B kichik model, 4-bit uni sezilarli yomonlashtirishi mumkin; 8-bit sifatni deyarli
saqlaydi. Sifatni tekshirish (68 savol): `python hybrid/2_engine/eval_api.py --url http://127.0.0.1:8080/v1 --name gguf-q8`.

### Ikkinchi model (ixtiyoriy): tayyor model bilan solishtirish

Chat sahifadagi ro'yxatda bizning model yonida **Qwen3.5-4B** ham turadi: o'sha Qwen oilasi, lekin kattaroq va Qwen
jamoasi to'liq post-training qilgan. Dars uchun: "o'zimiz o'qitgan 1.5B" va "tayyor olingan 4B" javoblarini yonma-yon
ko'rsatish. Fayl (2.7 GB, Apache 2.0):

```
curl -L -o hybrid/models/Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
```
Fayl bo'lsa, `run.py` uni ikkinchi llama-server'da (`:8081`) ishga tushiradi; bo'lmasa, ro'yxatda "o'chiq" bo'ladi.
API'da model nomi: `qwen3.5-4b`. RAM: ikkala model birga taxminan 5 GB.

## 2. Birinchi ishga tushirish (oynada, loglarni ko'rish uchun)

```
python hybrid/5_runner/run.py --admin siz@gmail.com
```
1-2 daqiqada `TAYYOR` chiqadi. Skript yangi tunnel manzilini `hybrid/data/backend_link.txt` ga yozadi va
**teach-help.vercel.app ni avtomatik shu manzilga ulaydi** (vercel CLI login bo'lsa). To'xtatish: Ctrl+C.

## 3. Doimiy rejim

| Fayl | Nima qiladi |
|---|---|
| `5_runner/start.bat` | fon rejimida (oynasiz) ishga tushiradi |
| `5_runner/stop.bat` | hammasini to'xtatadi |
| `5_runner/install_autostart.bat` | Windows'ga kirganda o'zi ishga tushadi (Startup papkasi, admin kerak emas) + zaryadlovchida uxlash o'chadi |
| `5_runner/uninstall_autostart.bat` | yuqoridagini bekor qiladi |

Sozlamalar: `data/settings.env` (`ADMIN_EMAILS`, `SIGNUP_CODE`, `RATE_LIMIT`, `DAILY_LIMIT`, `PARALLEL`, `THREADS`,
`UPDATE_VERCEL`). O'zgartirgach: `stop.bat`, keyin `start.bat`. Loglar: `data/run.log`, `llama.log`, `qwen.log`, `gateway.log`,
`tunnel.log`. Baza: `data/teachhelp.db` (boshqa serverga ko'chirganda shu faylni olib o'tasiz).

## Cheklovlar

- Kompyuter o'chsa yoki uxlasa — server ham to'xtaydi (ma'lumotlar saqlanadi, yonganda davom etadi).
- Bepul tunnel manzili har ishga tushganda yangi: UI (teach-help.vercel.app) o'zi yangilanadi, lekin API
  ishlatayotgan dasturlarda `base_url` ni yangilash kerak (`data/backend_link.txt`). Doimiy manzil uchun:
  Cloudflare'da domen (~$10/yil) + named tunnel, yoki doimiy bulut server.
- CPU'da javob bir necha soniya; `PARALLEL` (standart 2) ta so'rov bir vaqtda, qolganlari navbatda.
