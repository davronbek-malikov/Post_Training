# Lokal server: o'z kompyuteringiz — TeachHelp API serveri (bepul)

Oracle bepul serveri band bo'lganda yoki sinash uchun: kompyuteringiz server bo'ladi. Foydalanuvchilar istalgan joydan
ro'yxatdan o'tib `sk-th-...` kalit yaratadi va API'ni ishlatadi, siz admin panelda kim qancha ishlatganini ko'rasiz.
Arxitektura Oracle bilan bir xil (`deploy/oracle/gateway.py` + llama.cpp), faqat internetga Cloudflare tunnel chiqaradi.

```
foydalanuvchi ──HTTPS──► Cloudflare tunnel ──► gateway (127.0.0.1:7860) ──► llama-server (127.0.0.1:8080)
                         (port ochish shart emas,  hisoblar, kalitlar,         GGUF model, CPU
                          IP manzil ko'rinmaydi)    limit, admin, /ui
```

## 1. Bir martalik tayyorgarlik

| Kerak | Qayerda | Qanday |
|---|---|---|
| GGUF model | `deploy/oracle/model/teachhelp-1.5b-q8_0.gguf` | [oracle/README.md](../oracle/README.md), 1-qadam |
| llama.cpp (Windows, CPU) | `deploy/local/bin/llama/llama-server.exe` | https://github.com/ggml-org/llama.cpp/releases -> `llama-<versiya>-bin-win-cpu-x64.zip` -> `bin/llama/` ga oching |
| cloudflared | `deploy/local/bin/cloudflared.exe` | https://github.com/cloudflare/cloudflared/releases -> `cloudflared-windows-amd64.exe` |
| Python kutubxonalari | | `pip install fastapi uvicorn httpx` |

`bin/` va `data/` git'ga tushmaydi.

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
| `install_autostart.bat` | Windows'ga kirganda o'zi ishga tushadi (Task Scheduler) + zaryadlovchida uxlash o'chadi |
| `uninstall_autostart.bat` | yuqoridagini bekor qiladi |

Sozlamalar: `data/settings.env` (`ADMIN_EMAILS`, `SIGNUP_CODE`, `RATE_LIMIT`, `DAILY_LIMIT`, `PARALLEL`, `THREADS`,
`UPDATE_VERCEL`). O'zgartirgach: `stop.bat`, keyin `start.bat`. Loglar: `data/run.log`, `llama.log`, `gateway.log`,
`tunnel.log`. Baza: `data/teachhelp.db` (Oracle'ga ko'chirganda shu faylni olib o'tasiz).

## Cheklovlar

- Kompyuter o'chsa yoki uxlasa — server ham to'xtaydi (ma'lumotlar saqlanadi, yonganda davom etadi).
- Bepul tunnel manzili har ishga tushganda yangi: UI (teach-help.vercel.app) o'zi yangilanadi, lekin API
  ishlatayotgan dasturlarda `base_url` ni yangilash kerak (`data/backend_link.txt`). Doimiy manzil uchun:
  Cloudflare'da domen (~$10/yil) + named tunnel, yoki Oracle server.
- CPU'da javob bir necha soniya; `PARALLEL` (standart 2) ta so'rov bir vaqtda, qolganlari navbatda.
