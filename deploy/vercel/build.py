"""Vercel uchun statik sahifa: chat_ui/index.html + results.json + config.js -> deploy/vercel/ustoz-chat/.

Ishga tushirish:  python deploy/vercel/build.py
                  cd deploy/vercel/ustoz-chat && vercel deploy --prod --yes
Backend manzili sahifa havolasida beriladi: https://<sayt>/?api=https://<tunnel-manzil> (lokal yoki Kaggle backend
ishga tushganda havolani o'zi chiqaradi). Sahifa so'rovlarni to'g'ridan-to'g'ri backendga yuboradi (app.py da CORS
ochiq): Vercel faqat statik fayllarni beradi, model ishlamaydi.
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
CHAT_UI = HERE.parent.parent / "chat_ui"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", default="", help="doimiy backend manzili (ixtiyoriy; odatda ?api=... ishlatiladi)")
    args = p.parse_args()
    dist = HERE / "ustoz-chat"   # Vercel loyiha nomini papka nomidan oladi
    dist.mkdir(exist_ok=True)
    for old in dist.iterdir():   # .vercel/ (loyihaga ulanish) qoladi: keyingi deploy o'sha saytni yangilaydi
        if old.name != ".vercel":
            shutil.rmtree(old) if old.is_dir() else old.unlink()
    for name in ("index.html", "results.json"):
        shutil.copy(CHAT_UI / name, dist)
    (dist / "config.js").write_text(f"window.API_BASE = {json.dumps(args.api.rstrip('/'))};\n", encoding="utf-8")
    print("Tayyor:", dist, "| backend:", args.api or "?api=... orqali")


if __name__ == "__main__":
    main()
