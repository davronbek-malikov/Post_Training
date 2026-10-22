"""Lokal ("enterprise") backend: model sizning kompyuteringizda, sahifa Vercel'da.

Ishga tushirish:  python deploy/local/start_backend.py      (yoki start_backend.bat ni ikki marta bosing)
To'xtatish:      Ctrl+C

Nima qiladi:
  1. cloudflared (Cloudflare tunnel) yo'q bo'lsa, rasmiy GitHub release'dan deploy/local/bin/ ga yuklab oladi.
  2. deploy/hf_space/app.py ni ishga tushiradi: Qwen2.5-1.5B + DPO v2 + RLHF (Gemini hakam), CPU'da (~20 s javob).
  3. Bepul tunnel ochadi (https://....trycloudflare.com) va tayyor havolani chiqaradi:
       https://teach-help.vercel.app/?api=https://....trycloudflare.com
     Shu havolani ochgan har kim chat qila oladi, toki bu oyna ochiq tursa. Tunnel manzili har safar yangi.
"""
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
FRONTEND = "https://teach-help.vercel.app"
PORT = 7860
CLOUDFLARED = HERE / "bin" / ("cloudflared.exe" if os.name == "nt" else "cloudflared")
RELEASE = "https://github.com/cloudflare/cloudflared/releases/latest/download/" + (
    "cloudflared-windows-amd64.exe" if os.name == "nt" else "cloudflared-linux-amd64")


def ensure_cloudflared():
    if CLOUDFLARED.exists():
        return
    print("cloudflared yuklab olinmoqda (bir marta):", RELEASE, flush=True)
    CLOUDFLARED.parent.mkdir(exist_ok=True)
    urllib.request.urlretrieve(RELEASE, CLOUDFLARED)
    if os.name != "nt":
        CLOUDFLARED.chmod(0o755)


def wait_backend(proc, timeout=600):
    """Model yuklanguncha kutadi (CPU'da 1-2 daqiqa)."""
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit("Backend to'xtab qoldi: yuqoridagi xatoni qarang")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/api/models", timeout=5) as r:
                return [m["id"] for m in __import__("json").load(r)]
        except OSError:
            time.sleep(3)
    raise SystemExit("Backend 10 daqiqada ishga tushmadi")


def main():
    base = ROOT / "Qwen2.5-1.5B"
    if not (base / "model.safetensors").exists():
        raise SystemExit(f"Base model topilmadi: {base}")
    ensure_cloudflared()

    env = {**os.environ, "BASE_PATH": str(base), "ADAPTERS_PATH": str(ROOT / "chat_ui" / "adapters"),
           "PYTHONIOENCODING": "utf-8"}
    print("Backend ishga tushmoqda (model yuklanadi, ~1-2 daqiqa)...", flush=True)
    backend = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--host", "127.0.0.1", "--port", str(PORT)],
                               cwd=ROOT / "deploy" / "hf_space", env=env)
    tunnel = None
    try:
        models = wait_backend(backend)
        print("Modellar:", ", ".join(models), flush=True)
        tunnel = subprocess.Popen([str(CLOUDFLARED), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        url = None
        for line in tunnel.stdout:   # cloudflared manzilni o'z log'iga yozadi
            found = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if found:
                url = found.group(0)
                break
        if not url:
            raise SystemExit("Tunnel manzili olinmadi")
        print("\n" + "=" * 70)
        print("TAYYOR. Shu havolani oching yoki talabalarga yuboring:\n")
        print(f"   {FRONTEND}/?api={url}\n")
        print("Bu oyna ochiq turganda ishlaydi. To'xtatish: Ctrl+C")
        print("=" * 70 + "\n", flush=True)
        backend.wait()
    except KeyboardInterrupt:
        print("\nTo'xtatilmoqda...")
    finally:
        for proc in (tunnel, backend):
            if proc and proc.poll() is None:
                proc.terminate()


if __name__ == "__main__":
    main()
