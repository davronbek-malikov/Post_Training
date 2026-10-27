# Oracle Always Free: TeachHelp 24/7, bepul — UI, API kalitlar, admin panel

Kaggle backend dars uchun (GPU, vaqtincha). Bu papka esa **doimiy** server uchun: Oracle'ning bepul ARM serverida
(4 CPU, 24 GB RAM) model CPU'da ishlaydi, foydalanuvchilar ro'yxatdan o'tib `sk-th-...` kalit yaratadi, siz esa admin
panelda kim qancha ishlatganini ko'rasiz. Model qayta o'qitilmaydi: tayyor DPO v2 boshqa formatga o'tkaziladi xolos.

```
foydalanuvchi / dastur ──HTTPS──► Caddy ──► gateway.py (:7860)                ──► llama-server (:8080)
                                             hisoblar, kalitlar, limit, sarf,        GGUF model (8-bit), CPU,
                                             admin panel, /ui chat sahifa            4 ta so'rov parallel
```

| Fayl | Vazifasi |
|---|---|
| `merge.py` | DPO v2 LoRA adapterini asl Qwen2.5-1.5B ga qo'shadi -> bitta model (`model/teachhelp-1.5b/`) |
| `gateway.py` | API gateway: `deploy/hf_space/accounts.py` + `/v1` (OpenAI formati), model so'rovlarini llama-server'ga uzatadi. torch kerak emas |
| `setup.sh` | Serverni bitta buyruq bilan sozlaydi: llama.cpp, model, gateway, HTTPS, systemd, kunlik zaxira |
| `eval_api.py` | Istalgan OpenAI formatidagi API'ni 68 ta test savoli bilan baholaydi (GGUF 44/68 ni saqladimi?) |

**Nega vLLM emas?** Oracle'ning bepul serverida GPU yo'q. vLLM GPU uchun, CPU'da (ayniqsa ARM'da) llama.cpp ancha tez
va oson. Gateway har ikkalasi bilan bir xil OpenAI formatida gaplashadi: GPU server olinsa, `ENGINE_URL` ni vLLM'ga
qaratish kifoya — kalitlar, UI, admin o'zgarmaydi.

## 1. Modelni tayyorlash (o'z kompyuteringizda, bir marta)

```
python deploy/oracle/merge.py                                   # -> deploy/oracle/model/teachhelp-1.5b/ (~3 GB)
git clone --depth 1 https://github.com/ggml-org/llama.cpp       # GGUF konvertori
pip install gguf
python llama.cpp/convert_hf_to_gguf.py deploy/oracle/model/teachhelp-1.5b --outtype q8_0 \
       --outfile deploy/oracle/model/teachhelp-1.5b-q8_0.gguf   # ~1.6 GB
```
8-bit (q8_0) tanlangan: 1.5B kichik model, 4-bit uni sezilarli yomonlashtirishi mumkin; 8-bit sifatni deyarli
saqlaydi. `model/` git'ga tushmaydi (.gitignore).

## 2. Oracle'da server yaratish (Oracle konsoli)

**Compute -> Instances -> Create instance**
- Image: **Canonical Ubuntu 22.04**
- Shape: **Ampere -> VM.Standard.A1.Flex**, **4 OCPU, 24 GB** ("Always Free-eligible" yozuvi bo'lsin)
- Networking: standart, **public IPv4** bilan
- SSH keys: **Generate a key pair** -> private key'ni `C:\Users\<siz>\.ssh\oracle_teachhelp.key` ga saqlang
- "Out of capacity" chiqsa: keyinroq qayta urining yoki 2 OCPU / 12 GB tanlang

## 3. 80 va 443 portlarini ochish (Oracle konsoli)

Instance -> **Subnet** havolasi -> **Security** (yoki Security Lists) -> **Default Security List** ->
**Add Ingress Rules**: Source CIDR `0.0.0.0/0`, IP Protocol **TCP**, Destination Port Range **`80,443`**.
(Server ichidagi iptables'ni `setup.sh` o'zi ochadi.)

## 4. Modelni serverga ko'chirish va sozlash

```
scp -i ~/.ssh/oracle_teachhelp.key deploy/oracle/model/teachhelp-1.5b-q8_0.gguf ubuntu@<IP>:~
ssh -i ~/.ssh/oracle_teachhelp.key ubuntu@<IP>
curl -fsSL https://raw.githubusercontent.com/davronbek-malikov/Post_Training/main/deploy/oracle/setup.sh -o setup.sh
sudo ADMIN_EMAILS=siz@gmail.com MODEL_SRC=~/teachhelp-1.5b-q8_0.gguf bash setup.sh
```
~10-15 daqiqa (llama.cpp ARM uchun yig'iladi). Oxirida havolalar chiqadi:
`https://<ip-chiziqcha bilan>.sslip.io/ui`, API: `.../v1`. Ixtiyoriy sozlamalar: `SIGNUP_CODE` (taklif kodi),
`RATE_LIMIT` (kalit uchun so'rov/minut), `DAILY_LIMIT` (foydalanuvchi uchun so'rov/kun), `DOMAIN` (o'z domeningiz).

## 5. Tekshirish

```
python deploy/hf_space/test_api.py https://<domen>                                     # 8 bosqichli API testi
python deploy/oracle/eval_api.py --url https://<domen>/v1 --key sk-th-... --name oracle-gguf-q8   # 68 savol
```
Keyin `ADMIN_EMAILS` dagi email bilan ro'yxatdan o'ting: chap pastda **Admin** paydo bo'ladi.

## Boshqaruv (serverda)

| Buyruq | Nima qiladi |
|---|---|
| `systemctl status teachhelp-llama teachhelp-gateway caddy` | holat |
| `journalctl -u teachhelp-gateway -f` | gateway loglari (jonli) |
| `sudo nano /etc/teachhelp.env` -> `sudo systemctl restart teachhelp-gateway` | limit, admin, taklif kodini o'zgartirish |
| `cd /opt/teachhelp/Post_Training && sudo -u teachhelp git pull && sudo systemctl restart teachhelp-gateway` | kodni yangilash |
| `ls /var/lib/teachhelp/backups` | bazaning kunlik zaxira nusxalari (7 kun) |

## Narx

Hammasi Always Free: server (4 OCPU / 24 GB), 200 GB disk, oyiga 10 TB trafik. API so'rovlari uchun pul olinmaydi;
user ko'paysa faqat sekinlashadi (navbat). **Upgrade to Pay As You Go** tugmasini bosmang va faqat
"Always Free-eligible" resurslar yarating.
