"""Vercel uchun statik sahifa: chat_ui/index.html + results.json + config.js -> deploy/vercel/dist/.

Ishga tushirish:  python deploy/vercel/build.py --api https://<hf-foydalanuvchi>-<space-nomi>.hf.space
                  vercel deploy deploy/vercel/dist --prod
Sahifa so'rovlarni to'g'ridan-to'g'ri Hugging Face Space'ga yuboradi (app.py da CORS ochiq): Vercel faqat
statik fayllarni beradi, model ishlamaydi, shuning uchun bepul rejadagi vaqt chegaralari muammo emas.
"""
import argparse
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
CHAT_UI = HERE.parent.parent / "chat_ui"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api", required=True, help="Hugging Face Space manzili")
    args = p.parse_args()
    dist = HERE / "dist"
    shutil.rmtree(dist, ignore_errors=True)
    dist.mkdir()
    for name in ("index.html", "results.json"):
        shutil.copy(CHAT_UI / name, dist)
    (dist / "config.js").write_text(f"window.API_BASE = {json.dumps(args.api.rstrip('/'))};\n", encoding="utf-8")
    print("Tayyor:", dist, "->", args.api)


if __name__ == "__main__":
    main()
