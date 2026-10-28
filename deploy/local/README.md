# Lokal server: o'z kompyuteringiz — TeachHelp API serveri (bepul)

Kompyuteringiz server bo'ladi (hybrid: model va baza sizda, UI Vercel'da). Foydalanuvchilar istalgan joydan
ro'yxatdan o'tib `sk-th-...` kalit yaratadi va API'ni ishlatadi, siz admin panelda kim qancha ishlatganini ko'rasiz.
Model qayta o'qitilmaydi: tayyor DPO v2 GGUF formatiga o'tkaziladi va llama.cpp bilan CPU'da ishlaydi.

```
foydalanuvchi ──HTTPS──► Cloudflare tunnel ──► gateway (127.0.0.1:7860) ──► llama-server (127.0.0.1:8080)
                         (port ochish shart emas,  hisoblar, kalitlar,         GGUF model, CPU
                          IP manzil ko'rinmaydi)    limit, admin, /ui
```

## 1. Bir martalik tayyorgarlik

| Kerak | Qayerda | Qanday |
|---|---|---|
| GGUF model | `deploy/local/models/teachhelp-1.5b-q8_0.gguf` | pastda, "Modelni tayyorlash" |
| llama.cpp (Windows, CPU) | `deploy/local/bin/llama/llama-server.exe` | https://github.com/ggml-org/llama.cpp/releases -> `llama-<versiya>-bin-win-cpu-x64.zip` -> `bin/llama/` ga oching |
| cloudflared | `deploy/local/bin/cloudflared.exe` | https://github.com/cloudflare/cloudflared/releases -> `cloudflared-windows-amd64.exe` |
| Python kutubxonalari | | `pip install fastapi uvicorn httpx` |

`bin/`, `data/` va `models/` git'ga tushmaydi.

### Modelni tayyorlash (bir marta)

```
python deploy/local/merge.py                                    # LoRA + base -> models/teachhelp-1.5b/ (~3 GB)
git clone --depth 1 https://github.com/ggml-org/llama.cpp       # GGUF konvertori
pip install gguf
python llama.cpp/convert_hf_to_gguf.py deploy/local/models/teachhelp-1.5b --outtype q8_0 \
       --outfile deploy/local/models/teachhelp-1.5b-q8_0.gguf   # ~1.6 GB
```
8-bit (q8_0) tanlangan: 1.5B kichik model, 4-bit uni sezilarli yomonlashtirishi mumkin; 8-bit sifatni deyarli
saqlaydi. Sifatni tekshirish (68 savol): `python deploy/local/eval_api.py --url http://127.0.0.1:8080/v1 --name gguf-q8`.

### Ikkinchi model (ixtiyoriy): tayyor model bilan solishtirish

Chat sahifadagi ro'yxatda bizning model yonida **Qwen3.5-4B** ham turadi: o'sha Qwen oilasi, lekin kattaroq va Qwen
jamoasi to'liq post-training qilgan. Dars uchun: "o'zimiz o'qitgan 1.5B" va "tayyor olingan 4B" javoblarini yonma-yon
ko'rsatish. Fayl (2.7 GB, Apache 2.0):

```
curl -L -o deploy/local/models/Qwen3.5-4B-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf
```
Fayl bo'lsa, `run.py` uni ikkinchi llama-server'da (`:8081`) ishga tushiradi; bo'lmasa, ro'yxatda "o'chiq" bo'ladi.
API'da model nomi: `qwen3.5-4b`. RAM: ikkala model birga taxminan 5 GB.

## 2. Birinchi ishga tushirish (oynada, loglarni ko'rish uchun)

```
python deploy/local/run.py --admin siz@gmail.com
```
1-2 daqiqada `TAYYOR` chiqadi. Skript yangi tunnel manzilini `data/backend_link.txt` ga yozadi va
**teach-help.vercel.app ni avtomatik shu manzilga ulaydi** (vercel CLI login bo'lsa). To'xtatish: Ctrl+C.

## 3. Doimiy rejim

| Fayl | Nima qiladi |
|---|---|
| `start.bat` | fon rejimida (oynasiz) ishga tushiradi |
| `stop.bat` | hammasini to'xtatadi |
| `install_autostart.bat` | Windows'ga kirganda o'zi ishga tushadi (Startup papkasi, admin kerak emas) + zaryadlovchida uxlash o'chadi |
| `uninstall_autostart.bat` | yuqoridagini bekor qiladi |

Sozlamalar: `data/settings.env` (`ADMIN_EMAILS`, `SIGNUP_CODE`, `RATE_LIMIT`, `DAILY_LIMIT`, `PARALLEL`, `THREADS`,
`UPDATE_VERCEL`). O'zgartirgach: `stop.bat`, keyin `start.bat`. Loglar: `data/run.log`, `llama.log`, `qwen.log`, `gateway.log`,
`tunnel.log`. Baza: `data/teachhelp.db` (boshqa serverga ko'chirganda shu faylni olib o'tasiz).

## Cheklovlar

- Kompyuter o'chsa yoki uxlasa — server ham to'xtaydi (ma'lumotlar saqlanadi, yonganda davom etadi).
- Bepul tunnel manzili har ishga tushganda yangi: UI (teach-help.vercel.app) o'zi yangilanadi, lekin API
  ishlatayotgan dasturlarda `base_url` ni yangilash kerak (`data/backend_link.txt`). Doimiy manzil uchun:
  Cloudflare'da domen (~$10/yil) + named tunnel, yoki doimiy bulut server.
- CPU'da javob bir necha soniya; `PARALLEL` (standart 2) ta so'rov bir vaqtda, qolganlari navbatda.
