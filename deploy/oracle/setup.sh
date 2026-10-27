#!/usr/bin/env bash
# TeachHelp: Oracle Always Free serverini (Ubuntu 22.04, ARM) bitta buyruq bilan sozlash.
#
#   ssh ubuntu@<IP>
#   curl -fsSL https://raw.githubusercontent.com/davronbek-malikov/Post_Training/main/deploy/oracle/setup.sh -o setup.sh
#   sudo ADMIN_EMAILS=siz@gmail.com MODEL_SRC=~/teachhelp-1.5b-q8_0.gguf bash setup.sh   (yoki HF_TOKEN=hf_... bilan HF repodan)
#
# Natija (hammasi systemd'da, server qayta yonganda o'zi ishga tushadi):
#   internet ──443──► Caddy (HTTPS, bepul Let's Encrypt sertifikat)
#                       └──► gateway (127.0.0.1:7860): hisoblar, sk-th-... kalitlar, limit, sarf, admin, /ui
#                              └──► llama-server (127.0.0.1:8080): GGUF model, CPU, 4 ta so'rov parallel
# Domen: DOMAIN berilmasa <ip>.sslip.io (ro'yxatdan o'tmasdan ishlaydigan bepul DNS: IP'ni o'zi qaytaradi).
# Qayta ishga tushirish xavfsiz: har qadam bor narsani tekshiradi.
set -euo pipefail

ADMIN_EMAILS="${ADMIN_EMAILS:?ADMIN_EMAILS kerak: admin panelga kiradigan email}"
MODEL_REPO="${MODEL_REPO:-davronbekmalikov/teachhelp-1.5b-gguf}"   # HF'dagi shaxsiy repo
MODEL_FILE="${MODEL_FILE:-teachhelp-1.5b-q8_0.gguf}"
HF_TOKEN="${HF_TOKEN:-}"            # shaxsiy repo uchun (read huquqi yetadi)
# MODEL_SRC=/home/ubuntu/teachhelp-1.5b-q8_0.gguf — model scp bilan ko'chirilgan bo'lsa, HF kerak emas
SIGNUP_CODE="${SIGNUP_CODE:-}"      # taklif kodi (bo'sh — hamma ro'yxatdan o'ta oladi)
RATE_LIMIT="${RATE_LIMIT:-20}"      # kalit uchun so'rov/minut
DAILY_LIMIT="${DAILY_LIMIT:-500}"   # foydalanuvchi uchun so'rov/kun
PARALLEL="${PARALLEL:-4}"           # llama-server bir vaqtda nechta so'rovga javob yozadi
REPO_URL="${REPO_URL:-https://github.com/davronbek-malikov/Post_Training.git}"

APP=/opt/teachhelp            # kod, llama.cpp, model
DATA=/var/lib/teachhelp       # baza va zaxira nusxalar
IP="$(curl -fsS https://api.ipify.org)"
DOMAIN="${DOMAIN:-${IP//./-}.sslip.io}"

step() { echo; echo "=== $* ==="; }

step "1/8 Paketlar"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q build-essential cmake git curl python3-venv python3-pip sqlite3 \
    debian-keyring debian-archive-keyring apt-transport-https iptables-persistent

step "2/8 Foydalanuvchi va papkalar"
id teachhelp &>/dev/null || useradd --system --home "$APP" --shell /usr/sbin/nologin teachhelp
mkdir -p "$APP/models" "$DATA/backups"

step "3/8 Kod (GitHub)"
if [ -d "$APP/Post_Training/.git" ]; then
    git -C "$APP/Post_Training" pull -q
else
    GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 "$REPO_URL" "$APP/Post_Training"
fi
python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q -U pip
"$APP/venv/bin/pip" install -q fastapi uvicorn httpx pydantic

step "4/8 llama.cpp (ARM uchun manba koddan yig'iladi, ~5-10 daqiqa)"
if [ ! -x "$APP/llama.cpp/build/bin/llama-server" ]; then
    [ -d "$APP/llama.cpp" ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$APP/llama.cpp"
    cmake -S "$APP/llama.cpp" -B "$APP/llama.cpp/build" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF
    cmake --build "$APP/llama.cpp/build" --target llama-server -j "$(nproc)"
fi

step "5/8 Model: $MODEL_REPO/$MODEL_FILE"
if [ -n "${MODEL_SRC:-}" ] && [ ! -s "$APP/models/$MODEL_FILE" ]; then   # scp bilan ko'chirilgan fayl (HF'siz)
    cp "$MODEL_SRC" "$APP/models/$MODEL_FILE"
fi
if [ ! -s "$APP/models/$MODEL_FILE" ]; then
    auth=()
    [ -n "$HF_TOKEN" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
    curl -fL --retry 3 "${auth[@]}" "https://huggingface.co/$MODEL_REPO/resolve/main/$MODEL_FILE" \
        -o "$APP/models/$MODEL_FILE.part"
    mv "$APP/models/$MODEL_FILE.part" "$APP/models/$MODEL_FILE"
fi
ls -lh "$APP/models/$MODEL_FILE"
chown -R teachhelp:teachhelp "$APP" "$DATA"

step "6/8 Servislar (systemd)"
cat > /etc/teachhelp.env <<EOF
ENGINE_URL=http://127.0.0.1:8080
DB_PATH=$DATA/teachhelp.db
ADMIN_EMAILS=$ADMIN_EMAILS
SIGNUP_CODE=$SIGNUP_CODE
RATE_LIMIT=$RATE_LIMIT
DAILY_LIMIT=$DAILY_LIMIT
EOF
chmod 640 /etc/teachhelp.env
chgrp teachhelp /etc/teachhelp.env

cat > /etc/systemd/system/teachhelp-llama.service <<EOF
[Unit]
Description=TeachHelp model (llama.cpp server)
After=network.target

[Service]
User=teachhelp
ExecStart=$APP/llama.cpp/build/bin/llama-server -m $APP/models/$MODEL_FILE --host 127.0.0.1 --port 8080 \
    --ctx-size $((PARALLEL * 1024)) --parallel $PARALLEL --threads $(nproc) --jinja
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

cat > /etc/systemd/system/teachhelp-gateway.service <<EOF
[Unit]
Description=TeachHelp API gateway (accounts, API keys, limits, usage)
After=network.target teachhelp-llama.service

[Service]
User=teachhelp
EnvironmentFile=/etc/teachhelp.env
WorkingDirectory=$APP/Post_Training/deploy/oracle
ExecStart=$APP/venv/bin/uvicorn gateway:app --host 127.0.0.1 --port 7860 --proxy-headers
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now teachhelp-llama teachhelp-gateway
systemctl restart teachhelp-llama teachhelp-gateway

step "7/8 HTTPS: Caddy + $DOMAIN"
if ! command -v caddy &>/dev/null; then
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
    curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
    apt-get update -q && apt-get install -y -q caddy
fi
cat > /etc/caddy/Caddyfile <<EOF
$DOMAIN {
    encode gzip
    reverse_proxy 127.0.0.1:7860 {
        flush_interval -1
    }
}
EOF
# Oracle Ubuntu image'ida iptables 22-portdan boshqasini yopadi: 80 va 443 ochiladi
for port in 80 443; do
    iptables -C INPUT -p tcp --dport $port -j ACCEPT 2>/dev/null || iptables -I INPUT -p tcp --dport $port -j ACCEPT
done
netfilter-persistent save
systemctl restart caddy

step "8/8 Bazaning kunlik zaxira nusxasi (7 kun saqlanadi)"
cat > /etc/cron.daily/teachhelp-backup <<EOF
#!/bin/sh
sqlite3 $DATA/teachhelp.db ".backup '$DATA/backups/teachhelp-\$(date +%F).db'"
find $DATA/backups -name 'teachhelp-*.db' -mtime +7 -delete
EOF
chmod +x /etc/cron.daily/teachhelp-backup

step "Tekshiruv"
for i in $(seq 1 60); do
    curl -fsS http://127.0.0.1:8080/health >/dev/null 2>&1 && break
    sleep 2
done
curl -fsS http://127.0.0.1:7860/ && echo
echo
echo "======================================================================"
echo " TAYYOR (HTTPS sertifikat 1-2 daqiqada olinadi)"
echo "   UI               : https://$DOMAIN/ui"
echo "   Vercel UI        : https://teach-help.vercel.app/?api=https://$DOMAIN"
echo "   API base URL     : https://$DOMAIN/v1"
echo "   API hujjati      : https://$DOMAIN/docs"
echo "   Admin            : $ADMIN_EMAILS bilan ro'yxatdan o'ting -> chap pastda 'Admin'"
echo " Oracle konsolida 80 va 443 portlari ochiq bo'lishi kerak (README, 3-qadam)."
echo "======================================================================"
