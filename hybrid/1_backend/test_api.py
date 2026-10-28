"""API'ni boshidan oxirigacha sinash: foydalanuvchi yaratish -> kalit -> /v1 so'rovlar -> kalitni bekor qilish.

Ishga tushirish:  python hybrid/1_backend/test_api.py http://127.0.0.1:7860
                  python hybrid/1_backend/test_api.py https://xxx.trycloudflare.com   (internet orqali)
Foydalanuvchi UI'da qiladigan ishlarni (ro'yxatdan o'tish, kalit yaratish) shu yerda HTTP orqali qiladi, keyin
kalit bilan dastur kabi so'rov yuboradi. Faqat stdlib; `openai` o'rnatilgan bo'lsa, u bilan ham sinaydi.
"""
import json
import secrets
import sys
import time
import urllib.error
import urllib.request

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:7860").rstrip("/")
SIGNUP_CODE = sys.argv[2] if len(sys.argv) > 2 else ""


def call(method, path, body=None, token=None):
    req = urllib.request.Request(URL + path, method=method, data=json.dumps(body).encode() if body is not None else None,
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + token} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read().decode()
            return r.status, (json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw), r.headers
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}"), e.headers


def check(name, ok, extra=""):
    print(("  OK   " if ok else "  XATO ") + name, extra)
    if not ok:
        sys.exit(1)


print("Server:", URL)
email = f"test-{secrets.token_hex(3)}@example.com"
s, r, _ = call("POST", "/auth/signup", {"email": email, "password": "parol12345", "code": SIGNUP_CODE})
check("1. ro'yxatdan o'tish", s == 200, email)
session = r["token"]
check("   o'sha email ikkinchi marta -> 409", call("POST", "/auth/signup", {"email": email, "password": "parol12345", "code": SIGNUP_CODE})[0] == 409)
check("   noto'g'ri parol -> 401", call("POST", "/auth/login", {"email": email, "password": "xato"})[0] == 401)

s, r, _ = call("POST", "/keys", {"name": "test"}, session)
check("2. API kalit yaratish", s == 200 and r["key"].startswith("sk-th-"), r.get("preview", ""))
key = r["key"]
keys = call("GET", "/keys", token=session)[1]
check("   ro'yxatda faqat qisqa ko'rinishi (kalitning o'zi yo'q)", key not in json.dumps(keys) and len(keys) == 1)

check("3. kalitsiz /v1/models -> 401", call("GET", "/v1/models")[0] == 401)
check("   noto'g'ri kalit -> 401", call("GET", "/v1/models", token="sk-th-yolgon")[0] == 401)
s, r, _ = call("GET", "/v1/models", token=key)
check("   kalit bilan /v1/models", s == 200, [m["id"] for m in r["data"]])
model = r["data"][0]["id"]

t = time.time()
s, r, h = call("POST", "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": "Python nima?"}]}, key)
check("4. /v1/chat/completions", s == 200 and r["choices"][0]["message"]["content"].strip() != "",
      f"({time.time() - t:.1f} s, {r.get('usage')}, qolgan so'rovlar: {h.get('x-ratelimit-remaining-requests')})")
print("       javob:", r["choices"][0]["message"]["content"][:150].replace("\n", " "))

t = time.time()
s, raw, _ = call("POST", "/v1/chat/completions", {"model": model, "stream": True, "stream_options": {"include_usage": True},
                                                  "messages": [{"role": "user", "content": "What is 2 + 2 * 3?"}]}, key)
events = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ") and line != "data: [DONE]"]
text = "".join(e["choices"][0]["delta"].get("content", "") for e in events if e["choices"])
check("5. stream (SSE)", s == 200 and raw.rstrip().endswith("data: [DONE]") and text.strip() != "",
      f"({time.time() - t:.1f} s, {len(events)} bo'lak, usage: {events[-1].get('usage')})")
print("       javob:", text[:150].replace("\n", " "))

try:
    from openai import OpenAI
    client = OpenAI(base_url=URL + "/v1", api_key=key)
    out = client.chat.completions.create(model=model, messages=[{"role": "user", "content": "Overfitting nima?"}])
    check("6. openai kutubxonasi bilan", out.choices[0].message.content.strip() != "", out.usage)
except ImportError:
    print("  --   6. openai o'rnatilmagan, o'tkazib yuborildi")

usage = call("GET", "/usage", token=session)[1]
check("7. sarf (usage) yozildi", usage and usage[0]["requests"] >= 2, usage)

call("DELETE", f"/keys/{keys[0]['id']}", token=session)
check("8. bekor qilingan kalit -> 401", call("GET", "/v1/models", token=key)[0] == 401)
print("Hammasi o'tdi.")
