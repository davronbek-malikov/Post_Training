# 3. UI — Vercel

UI oddiy HTML/JS sahifa: brauzer uni Vercel'dan yuklab oladi, keyin so'rovlarni to'g'ridan-to'g'ri backend'ga
(tunnel manziliga) yuboradi. Vercel'da model ishlamaydi, u faqat fayllarni beradi.

| Fayl | Nima qiladi |
|---|---|
| [chat_ui/index.html](../../chat_ui/index.html) | Sahifaning o'zi: chat (ikki model, nusxalash), API kalitlar oynasi (model va muddat tanlash, narxlar, sarf, kod namunalari), admin panel. Bitta joyda turadi: Kaggle darsi ham shu faylni ishlatadi |
| [build.py](build.py) | `index.html` + `results.json` ni `teach-help/` ga nusxalaydi va backend manzilini `config.js` ga yozadi |
| `teach-help/` | Yig'ilgan sahifa, Vercel'ga shu papka yuklanadi (git'da yo'q) |

UI backend manzilini qanday biladi:

```
build.py --api https://xxx.trycloudflare.com   ->   teach-help/config.js:  window.API_BASE = "https://xxx..."
index.html  ->  fetch(API_BASE + "/v1/...")
```

`config.js` keshlanmaydi (`vercel.json`), shuning uchun tunnel manzili o'zgarsa, brauzer darhol yangisini oladi.
Tunnel manzili o'zgarganda [5_runner/run.py](../5_runner/run.py) `build.py` va `vercel deploy` ni o'zi bajaradi.

Qo'lda: `python hybrid/3_ui/build.py --api https://<tunnel>` va `cd hybrid/3_ui/teach-help && vercel deploy --prod --yes`.
