"""5. Hammasini bog'lovchi: o'z kompyuteringiz — TeachHelp serveri (model + API kalitlar + admin), Cloudflare tunnel orqali internetda.

  internet ──► Cloudflare tunnel ──► gateway (127.0.0.1:7860) ──► llama-server (127.0.0.1:8080, GGUF, CPU)  bizning model
              (cloudflared.exe)      1_backend/gateway.py       └─► llama-server (127.0.0.1:8081)             Qwen3.5-4B
                                                                     (models/ papkasida fayl bo'lsa)

Bu skript uchala jarayonni ishga tushiradi va kuzatib turadi: biri to'xtasa, qayta yoqadi. Tunnel manzili har
ishga tushganda yangi bo'ladi (bepul "quick tunnel"), shuning uchun skript yangi manzilni data/backend_link.txt ga
yozadi va teach-help.vercel.app sahifasini avtomatik shu manzilga ulaydi (UI havolasi o'zgarmaydi).

Birinchi marta:  python hybrid/5_runner/run.py --admin siz@gmail.com     (sozlamalar hybrid/data/settings.env ga yoziladi)
Keyingi safar:   python hybrid/5_runner/run.py            yoki   hybrid\\5_runner\\start.bat
Avtomatik ishga tushish (kompyuter yonganda): hybrid\\5_runner\\install_autostart.bat
"""
import argparse
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
HYBRID = HERE.parent            # hybrid/: 1_backend, 3_ui, bin, data, models
BIN, DATA = HYBRID / "bin", HYBRID / "data"
SETTINGS = DATA / "settings.env"
MODEL = HYBRID / "models" / "teachhelp-1.5b-q8_0.gguf"   # 2_engine/merge.py + GGUF konvertor (README, 2-qism)
# Solishtirish uchun tayyor model (ixtiyoriy): hybrid/README.md, "Ikkinchi model"
QWEN_MODEL = HYBRID / "models" / "Qwen3.5-4B-Q4_K_M.gguf"
DEFAULTS = {"ADMIN_EMAILS": "", "SIGNUP_CODE": "", "RATE_LIMIT": "20", "DAILY_LIMIT": "500",
            "PARALLEL": "2", "THREADS": str(max(1, (os.cpu_count() or 4) - 2)), "UPDATE_VERCEL": "1"}
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # pythonw bilan ishlaganda qora oynalar ochilmasin


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(map(str, a))
    print(line, flush=True)
    with open(DATA / "run.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_settings(admin):
    s = dict(DEFAULTS)
    if SETTINGS.exists():
        for line in SETTINGS.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                s[k.strip()] = v.strip()
    if admin:
        s["ADMIN_EMAILS"] = admin
    SETTINGS.write_text("# TeachHelp lokal server sozlamalari (o'zgartirgach run.py ni qayta ishga tushiring)\n"
                        + "".join(f"{k}={v}\n" for k, v in s.items()), encoding="utf-8")
    return s


def port_busy(port):
    with socket.socket() as sock:
        return sock.connect_ex(("127.0.0.1", port)) == 0


def wait_http(url, seconds):
    for _ in range(seconds):
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except OSError:
            time.sleep(1)
    return False


class Proc:
    """Bitta jarayon: ishga tushirish, log faylga yozish, to'xtaganini aniqlash."""

    def __init__(self, name, cmd, cwd=None, env=None, on_line=None):
        self.name, self.cmd, self.cwd, self.env, self.on_line, self.p = name, cmd, cwd, env, on_line, None

    def start(self):
        logf = open(DATA / f"{self.name}.log", "a", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen(self.cmd, cwd=self.cwd, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)

        def pump():   # chiqishni log faylga yozadi va kerak bo'lsa qatorlarni tekshiradi (tunnel manzili)
            for line in self.p.stdout:
                logf.write(line)
                logf.flush()
                if self.on_line:
                    self.on_line(line)
        threading.Thread(target=pump, daemon=True).start()
        log(self.name, "ishga tushdi (pid", f"{self.p.pid})")

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def stop(self):
        if self.alive():
            self.p.terminate()
            try:
                self.p.wait(10)
            except subprocess.TimeoutExpired:
                self.p.kill()


def publish(url, settings):
    """Yangi tunnel manzili: havolani faylga yozish va Vercel'dagi sahifani shu manzilga ulash."""
    (DATA / "backend_link.txt").write_text(
        f"UI:           https://teach-help.vercel.app   (yoki {url}/ui)\n"
        f"API base URL: {url}/v1\nAPI hujjati:  {url}/docs\nVaqt:         {time.ctime()}\n", encoding="utf-8")
    log("TAYYOR. UI: https://teach-help.vercel.app | lokal UI:", url + "/ui", "| API:", url + "/v1")
    if settings["UPDATE_VERCEL"] != "1":
        return
    vercel = shutil.which("vercel")
    if not vercel:
        log("vercel CLI topilmadi: sahifa avtomatik yangilanmadi. Havola:", url)
        return
    try:
        subprocess.run([sys.executable, str(HYBRID / "3_ui" / "build.py"), "--api", url],
                       check=True, capture_output=True, creationflags=NO_WINDOW)
        out = subprocess.run([vercel, "deploy", "--prod", "--yes"], cwd=HYBRID / "3_ui" / "teach-help",
                             capture_output=True, text=True, encoding="utf-8", errors="replace", creationflags=NO_WINDOW)
        log("Vercel:", "yangilandi -> teach-help.vercel.app endi shu serverga ulanadi" if out.returncode == 0
            else "xato: " + out.stdout[-300:] + out.stderr[-300:])
    except Exception as e:
        log("Vercel yangilanmadi:", repr(e))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--admin", help="admin panelga kiradigan email(lar), vergul bilan (bir marta beriladi)")
    p.add_argument("--no-tunnel", action="store_true", help="faqat lokal (internetga chiqmasdan) sinash uchun")
    args = p.parse_args()
    DATA.mkdir(exist_ok=True)
    s = load_settings(args.admin)
    if not s["ADMIN_EMAILS"]:
        sys.exit("Birinchi marta: python hybrid/5_runner/run.py --admin siz@gmail.com")
    for f in (MODEL, BIN / "llama" / "llama-server.exe", BIN / "cloudflared.exe"):
        if not f.exists():
            sys.exit(f"Topilmadi: {f}  (hybrid/README.md)")
    if port_busy(7860) or port_busy(8080) or port_busy(8081):
        sys.exit("7860, 8080 yoki 8081 port band: server allaqachon ishlayapti (data/backend_link.txt) yoki boshqa dastur")

    env = {**os.environ, "ENGINE_URL": "http://127.0.0.1:8080", "QWEN_ENGINE_URL": "http://127.0.0.1:8081", "DB_PATH": str(DATA / "teachhelp.db"),
           "PYTHONIOENCODING": "utf-8", **{k: s[k] for k in ("ADMIN_EMAILS", "SIGNUP_CODE", "RATE_LIMIT", "DAILY_LIMIT")}}
    parallel = int(s["PARALLEL"])
    llama = Proc("llama", [str(BIN / "llama" / "llama-server.exe"), "-m", str(MODEL), "--host", "127.0.0.1",
                           "--port", "8080", "--ctx-size", str(parallel * 1024), "--parallel", str(parallel),
                           "--threads", s["THREADS"], "--jinja"])
    # thinking o'chiq (--reasoning-budget 0): CPU'da fikrlash qismi javobni bir necha daqiqaga cho'zadi
    qwen = Proc("qwen", [str(BIN / "llama" / "llama-server.exe"), "-m", str(QWEN_MODEL), "--host", "127.0.0.1",
                         "--port", "8081", "--ctx-size", str(parallel * 4096), "--parallel", str(parallel),
                         "--threads", s["THREADS"], "--jinja", "--reasoning-budget", "0"])
    gateway = Proc("gateway", [sys.executable, "-m", "uvicorn", "gateway:app", "--host", "127.0.0.1", "--port", "7860"],
                   cwd=HYBRID / "1_backend", env=env)
    found = {}

    def on_tunnel_line(line):
        m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
        if m and found.get("url") != m.group(0):
            found["url"] = m.group(0)
    tunnel = Proc("tunnel", [str(BIN / "cloudflared.exe"), "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:7860"],
                  on_line=on_tunnel_line)
    engines = [llama] + ([qwen] if QWEN_MODEL.exists() else [])
    procs = engines + [gateway] + ([] if args.no_tunnel else [tunnel])

    log("=== TeachHelp lokal server | admin:", s["ADMIN_EMAILS"], "| parallel:", parallel, "| threads:", s["THREADS"])
    if not QWEN_MODEL.exists():
        log("Ikkinchi model yo'q (ixtiyoriy):", QWEN_MODEL)
    for engine, port in zip(engines, (8080, 8081)):
        engine.start()
        if not wait_http(f"http://127.0.0.1:{port}/health", 180):
            log(engine.name, f"3 daqiqada tayyor bo'lmadi: data/{engine.name}.log ni ko'ring")
    gateway.start()
    wait_http("http://127.0.0.1:7860/", 30)
    if not args.no_tunnel:
        tunnel.start()
    else:
        log("TAYYOR (faqat lokal): http://127.0.0.1:7860/ui")

    published, fails, tick = None, 0, 0
    try:
        while True:
            if found.get("url") and found["url"] != published:
                published, fails = found["url"], 0
                wait_http(published + "/", 60)   # yangi manzil DNS'da paydo bo'lguncha
                publish(published, s)
            tick += 1
            # Jarayon tirik bo'lsa ham tunnel o'lgan bo'lishi mumkin ("Tunnel not found": kompyuter uxlagan yoki
            # internet uzilgan). Har daqiqada manzilni internet orqali tekshiramiz; 3 marta ochilmasa — qayta yoqamiz.
            if published and not args.no_tunnel and tick % 4 == 0:
                fails = 0 if wait_http(published + "/", 1) else fails + 1
                if fails >= 3:
                    log("tunnel internetdan ochilmayapti -> yangi tunnel ochilyapti")
                    tunnel.stop()
                    fails = 0
            for proc in procs:
                if not proc.alive():
                    log(proc.name, "to'xtagan -> qayta ishga tushirilyapti")
                    if proc is tunnel:
                        found.pop("url", None)
                    proc.start()
            time.sleep(15)
    except KeyboardInterrupt:
        log("To'xtatilmoqda...")
    finally:
        for proc in reversed(procs):
            proc.stop()
        log("To'xtatildi.")


if __name__ == "__main__":
    main()
