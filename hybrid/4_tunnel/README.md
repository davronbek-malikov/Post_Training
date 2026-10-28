# 4. Tunnel — Cloudflare

Uydagi kompyuterga internetdan to'g'ridan-to'g'ri kirib bo'lmaydi: router (NAT), firewall va o'zgaruvchan IP.
Tunnel buni hal qiladi: `cloudflared.exe` kompyuterdan Cloudflare'ga **o'zi chiqib** ulanadi (chiquvchi ulanishni
router to'smaydi), Cloudflare esa ochiq manzil beradi va kelgan so'rovlarni shu ulanish orqali kompyuterga uzatadi.

```
1. cloudflared.exe (PC)  ── o'zi ulanadi ──►  Cloudflare   (manzil: https://xxx.trycloudflare.com)
2. foydalanuvchi  ──►  https://xxx.trycloudflare.com  ──►  Cloudflare  ──►  PC: gateway (127.0.0.1:7860)
```

Kod yozilmagan: [5_runner/run.py](../5_runner/run.py) shu buyruqni ishga tushiradi va chiqishidan manzilni o'qiydi:

```
bin/cloudflared.exe tunnel --no-autoupdate --url http://127.0.0.1:7860
```

| | Quick tunnel (bizda) | Named tunnel |
|---|---|---|
| Manzil | tasodifiy `xxx.trycloudflare.com` | doimiy, o'z domeningiz (`api.teachhelp.uz`) |
| Qayta yoqilganda | o'zgaradi | o'zgarmaydi |
| Kerak | hech narsa | Cloudflare hisobi + domen (~$10/yil) |

Afzalliklari: port ochish shart emas, IP manzil yashirin, HTTPS avtomatik, bepul. Tunnel o'lsa (kompyuter uxlagan
yoki internet uzilgan), `run.py` uni har daqiqada tekshiradi, qayta ochadi va yangi manzilni Vercel'ga yozadi.
Eslatma: shifrlash Cloudflare'da ochiladi, ya'ni so'rovlar Cloudflare orqali o'tadi.
