"""1. Backend: API gateway — hisoblar, sk-th-... kalitlar, limitlar, sarf — model esa llama.cpp'da.

Frontier API'lardagidek ikki qatlam:
  foydalanuvchi ──► gateway (shu fayl, :7860)  ──►  llama-server (llama.cpp, :8080, faqat ichki)
                    kalit, limit, sarf, admin        GGUF model, CPU'da, bir nechta so'rov parallel
API deploy/hf_space/app.py bilan bir xil (/api/*, /v1/*, /auth/*, /keys, /admin/*), shuning uchun chat_ui va
test_api.py o'zgarishsiz ishlaydi. Farqi: torch yo'q (yengil), model alohida jarayonda — gateway qayta ishga
tushsa ham model xotirada qoladi. Kelajakda GPU bo'lsa, ENGINE_URL ni vLLM serveriga qaratish kifoya.

Ikki model (dars uchun solishtirish): har biri o'z llama-server'ida, gateway "model" nomiga qarab yo'naltiradi.
  sft-lora-dpo-v2  — biz o'qitgan model (Qwen2.5-1.5B + SFT + LoRA + DPO v2)       ENGINE_URL       (:8080)
  qwen3.5-4b       — tayyor model: Qwen jamoasi to'liq post-training qilgan       QWEN_ENGINE_URL  (:8081)
Ikkinchi server ishlamasa, u model ro'yxatda "o'chiq" bo'lib turadi, birinchisi ishlayveradi.

Ishga tushirish:  ENGINE_URL=http://127.0.0.1:8080 uvicorn gateway:app --host 127.0.0.1 --port 7860
"""
import json
import os
import secrets
import sys
import time
from pathlib import Path

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))   # accounts.py shu papkada
from accounts import PRICES, RATE_LIMIT, ApiError, check_api_key, check_model, record_usage, router  # noqa: E402

CHAT_UI = HERE.parent.parent / "chat_ui"   # index.html (UI) bitta joyda: chat_ui/
STARTED = int(time.time())
# engine: llama-server yoki vLLM manzili. max_tokens: javob uzunligi chegarasi, ui_tokens: chat sahifadagi standart.
# sampling: shu model uchun engine'ga qo'shimcha yuboriladigan sozlamalar.
MODELS = [
    {"id": "sft-lora-dpo-v2", "label": "v3 · SFT + LoRA + DPO v2", "base": "1.5b", "group": "Preference optimisation",
     "desc": "Biz o'qitgan yakuniy model · 44/68", "desc_en": "Our final model, best score · 44/68",
     "base_model": "Qwen/Qwen2.5-1.5B", "license": "apache-2.0",
     "engine": os.environ.get("ENGINE_URL", "http://127.0.0.1:8080").rstrip("/"), "max_tokens": 300, "ui_tokens": 100,
     # Kaggle'dagi generatsiya bilan bir xil: greedy, repetition penalty 1.1 (test shunday o'tkazilgan).
     # repeat_last_n = bitta so'rovning butun konteksti (llama-server -1 ni qabul qilmaydi)
     "sampling": {"repeat_penalty": 1.1, "repeat_last_n": 1024}},
    {"id": "qwen3.5-4b", "label": "Qwen3.5-4B (tayyor model)", "base": "4b", "group": "Ready-made model",
     "desc": "Qwen jamoasi post-training qilgan, solishtirish uchun", "desc_en": "Post-trained by the Qwen team, for comparison",
     "base_model": "Qwen/Qwen3.5-4B", "license": "apache-2.0",
     "engine": os.environ.get("QWEN_ENGINE_URL", "http://127.0.0.1:8081").rstrip("/"), "max_tokens": 1024, "ui_tokens": 512,
     # "thinking" o'chiq: CPU'da fikrlash qismi juda uzoq; Qwen tavsiyasi bo'yicha takrorlanishga qarshi presence_penalty
     "sampling": {"chat_template_kwargs": {"enable_thinking": False}, "presence_penalty": 1.5}},
]
BY_ID = {m["id"]: m for m in MODELS}
PUBLIC = ("id", "label", "base", "group", "desc", "desc_en", "base_model", "license")   # /api/models da ko'rsatiladigan maydonlar

client = httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5))
app = FastAPI(title="TeachHelp API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)   # /auth/*, /keys, /usage, /admin/*


@app.exception_handler(ApiError)
def api_error(request, e):
    kind = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "invalid_request_error", 429: "rate_limit_error", 503: "overloaded_error"}.get(e.status, "api_error")
    return JSONResponse({"error": {"message": e.message, "type": kind, "code": e.code}}, status_code=e.status)


async def engine_ok(model):
    try:
        return (await client.get(model["engine"] + "/health", timeout=3)).status_code == 200
    except httpx.HTTPError:
        return False


def engine_body(model, question, max_tokens, temperature=0.0, top_p=1.0, stream=False):
    # Faqat bitta savol: bizning model bitta savol -> bitta javob ko'rinishida o'qitilgan.
    return {"messages": [{"role": "user", "content": question}], "max_tokens": max(1, min(max_tokens, model["max_tokens"])),
            "temperature": temperature, "top_p": top_p, "stream": stream, **model["sampling"],
            **({"stream_options": {"include_usage": True}} if stream else {})}


async def engine_open(model, body):
    """llama-server'ga so'rovni ochadi. Xato javob BOSHLANISHIDAN OLDIN tekshiriladi: aks holda brauzer faqat
    "Failed to fetch" ko'radi. Engine'ning xato matni gateway logiga yoziladi."""
    try:
        r = await client.send(client.build_request("POST", model["engine"] + "/v1/chat/completions", json=body), stream=True)
    except httpx.HTTPError as e:
        raise ApiError(503, "Model serveriga ulanib bo'lmadi", "engine_offline") from e
    if r.status_code != 200:
        detail = (await r.aread()).decode(errors="replace")[:300]
        await r.aclose()
        print("Engine xatosi:", model["id"], r.status_code, detail, flush=True)
        raise ApiError(503, f"Model serveri xato qaytardi ({r.status_code})", "engine_error")
    return r


async def engine_stream(r):
    """Ochilgan javobdan SSE bo'laklarini o'qiydi: ("text", bo'lak) va oxirida ("done", usage, finish_reason)."""
    usage, finish = {}, "stop"
    try:
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for c in chunk.get("choices", []):
                if c.get("delta", {}).get("content"):
                    yield "text", c["delta"]["content"]
                finish = c.get("finish_reason") or finish
    finally:
        await r.aclose()
    yield "done", usage, finish


@app.get("/")
async def health():
    status = {m["id"]: await engine_ok(m) for m in MODELS}
    return {"ok": True, "engine": status[MODELS[0]["id"]], "models": status, "api": "/v1", "docs": "/docs", "ui": "/ui"}


@app.get("/ui")
def ui():
    return FileResponse(CHAT_UI / "index.html")


@app.get("/results.json")
def results():
    return FileResponse(CHAT_UI / "results.json")


@app.get("/config.js")
def config():
    return Response("window.API_BASE = location.origin;", media_type="text/javascript")


# ---------- 1. UI: chat sahifa uchun (kalitsiz) ----------

class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = MODELS[0]["id"]
    messages: list[Message]
    max_new_tokens: int | None = None   # berilmasa — modelning ui_tokens qiymati


@app.get("/api/models")
async def models():
    out = []
    for m in MODELS:
        ok = await engine_ok(m)
        price_in, price_out = PRICES.get(m["id"], (0, 0))
        out.append({**{k: m[k] for k in PUBLIC}, "pretrained": False, "available": ok, "offline": not ok,
                    "price": {"input": price_in, "output": price_out}})   # $ / 1M token
    return out


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    if req.model not in BY_ID or not req.messages:
        raise HTTPException(400, "Noto‘g‘ri so‘rov")
    try:
        model = BY_ID[req.model]
        r = await engine_open(model, engine_body(model, req.messages[-1].content, req.max_new_tokens or model["ui_tokens"],
                                                 stream=True))
    except ApiError as e:
        raise HTTPException(503, e.message)   # UI xato matnini "detail" dan o'qiydi

    async def stream():
        async for kind, *data in engine_stream(r):
            if kind == "text":
                yield data[0]

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


# ---------- 2. API: dasturlar uchun, OpenAI formatida (API kalit bilan) ----------

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] = ""


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stream_options: dict | None = None


def text_of(content):
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content if p.get("type") == "text")


@app.get("/v1/models")
def v1_models(authorization: str = Header(""), x_api_key: str = Header("")):
    key, _ = check_api_key(authorization, x_api_key)
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": STARTED, "owned_by": "teachhelp",
                      "base_model": m["base_model"], "license": m["license"]} for m in MODELS
                     if not key["model"] or key["model"] == m["id"]]}


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest, authorization: str = Header(""), x_api_key: str = Header("")):
    key, remaining = check_api_key(authorization, x_api_key)
    if req.model not in BY_ID:
        raise ApiError(404, f"Model '{req.model}' topilmadi. Mavjud: {', '.join(BY_ID)}", "model_not_found")
    check_model(key, req.model)
    question = next((text_of(m.content) for m in reversed(req.messages) if m.role == "user"), "").strip()
    if not question:
        raise ApiError(400, "messages ichida 'user' xabari bo'lishi kerak", "invalid_request")
    model = BY_ID[req.model]
    body = engine_body(model, question, req.max_completion_tokens or req.max_tokens or model["max_tokens"],
                       req.temperature, req.top_p, True)
    r = await engine_open(model, body)   # model o'chiq yoki xato bo'lsa — shu yerda, OpenAI formatidagi 503
    cid, created, t0 = "chatcmpl-" + secrets.token_hex(12), int(time.time()), time.time()
    headers = {"x-ratelimit-limit-requests": str(RATE_LIMIT), "x-ratelimit-remaining-requests": str(remaining)}

    def usage_of(u):
        p, c = u.get("prompt_tokens", 0), u.get("completion_tokens", 0)
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}

    if not req.stream:
        parts, usage, finish = [], {}, "stop"
        async for kind, *data in engine_stream(r):
            if kind == "text":
                parts.append(data[0])
            else:
                usage, finish = data
        record_usage(key["id"], req.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                     int((time.time() - t0) * 1000))
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "".join(parts)}, "finish_reason": finish}],
            "usage": usage_of(usage)}, headers=headers)

    async def sse():
        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                                          "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]},
                                         ensure_ascii=False) + "\n\n"
        usage, finish = {}, "stop"
        try:
            yield chunk({"role": "assistant", "content": ""})
            async for kind, *data in engine_stream(r):
                if kind == "text":
                    yield chunk({"content": data[0]})
                else:
                    usage, finish = data
        finally:   # mijoz ulanishni uzsa ham, shu paytgacha bo'lgan sarf yoziladi
            record_usage(key["id"], req.model, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                         int((time.time() - t0) * 1000))
        yield chunk({}, finish)
        if (req.stream_options or {}).get("include_usage"):
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                         "model": req.model, "choices": [], "usage": usage_of(usage)}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)
