# 5. Hammasini bog'lovchi

[run.py](run.py) — "process supervisor": barcha qismlarni to'g'ri tartibda yoqadi va kuzatib turadi.

| Qadam | Nima qiladi |
|---|---|
| 1 | `data/settings.env` dan sozlamalarni o'qiydi (admin email, limitlar, parallel, threads) |
| 2 | `llama-server` ni yoqadi: bizning model `:8080`, Qwen3.5-4B `:8081` (fayli bo'lsa) |
| 3 | gateway'ni yoqadi: `uvicorn gateway:app` (`1_backend/` papkasida, `:7860`) |
| 4 | tunnel'ni yoqadi va chiqishidan `https://xxx.trycloudflare.com` manzilini topadi |
| 5 | manzilni `data/backend_link.txt` ga yozadi, `3_ui/build.py` + `vercel deploy` bilan UI'ni yangilaydi |
| 6 | har 15 soniyada: to'xtagan jarayonni qayta yoqadi; har daqiqada tunnel internetdan ochilishini tekshiradi |

| Fayl | Nima qiladi |
|---|---|
| [start.bat](start.bat) | `run.py` ni fon rejimida (oynasiz, `pythonw`) ishga tushiradi |
| [stop.bat](stop.bat) | `run.py`, llama-server, gateway va tunnel'ni to'xtatadi |
| [install_autostart.bat](install_autostart.bat) | Windows'ga kirganda server o'zi ishga tushadi (Startup papkasi), zaryadlovchida uxlash o'chadi |
| [uninstall_autostart.bat](uninstall_autostart.bat) | yuqoridagini bekor qiladi |

Loglar: `hybrid/data/run.log` (umumiy), `llama.log`, `qwen.log`, `gateway.log`, `tunnel.log`.
