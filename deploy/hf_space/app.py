"""TeachHelp backend: eng yaxshi modellar (Qwen2.5-1.5B + LoRA adapterlar), Kaggle GPU'da (CPU'da ham ishlaydi).

Ikki xil kirish yo'li, frontier modellardagidek:
  1. UI (chat sahifa): /api/models, /api/generate — chat_ui/server.py bilan bir xil, chat_ui/index.html shuni ishlatadi.
  2. API (dasturlar): /v1/models, /v1/chat/completions — OpenAI formatida, API kalit bilan (accounts.py).
     Shuning uchun tayyor `openai` kutubxonasi ishlaydi: OpenAI(base_url="https://.../v1", api_key="sk-th-...").
     Kalitlarni foydalanuvchilar UI'dagi "API kalitlar" menyusida o'zlari yaratadi (/auth/*, /keys — accounts.py).
Hujjat: <server>/docs (FastAPI avtomatik yaratadi).

Base model: asl Qwen/Qwen2.5-1.5B (adapterlar shu modelda o'qitilgan). Adapterlar: prepare.py (float16 nusxa).
Xotira: bitta base model va bir nechta kichik adapter. Tanlangan adapter base og'irliklarga qo'shib yuboriladi
(merge): generatsiya oddiy model tezligida. Boshqa model tanlansa, eskisi ayirib olinadi.
"""
import json
import os
import secrets
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

from accounts import RATE_LIMIT, ApiError, check_api_key, record_usage, router

HERE = Path(__file__).parent
# Lokal sinov uchun BASE_PATH / ADAPTERS_PATH bilan loyihadagi papkalarni berish mumkin.
BASE = Path(os.environ.get("BASE_PATH", HERE / "base"))              # snapshot_download("Qwen/Qwen2.5-1.5B")
ADAPTERS = Path(os.environ.get("ADAPTERS_PATH", HERE / "adapters"))  # prepare.py natijasi

# Faqat eng yaxshi modellar. adapters/<id> bo'lmasa, model ro'yxatga chiqmaydi.
MODELS = [
    {"id": "sft-lora-dpo-v2", "label": "v3 · SFT + LoRA + DPO v2",
     "desc": "Yakuniy model, eng yaxshi natija · 44/68", "desc_en": "Final model, best score · 44/68"},
    {"id": "sft-lora-rlhf-llm", "label": "v3 · SFT + LoRA + RLHF (Gemini judge)",
     "desc": "Eng yaxshi RLHF: Gemini API baholagan, RLOO · 41/68", "desc_en": "Best RLHF run: judged by Gemini API, RLOO · 41/68"},
]
MODELS = [{**m, "base": "1.5b", "group": "Preference optimisation", "pretrained": False, "available": True}
          for m in MODELS if (ADAPTERS / m["id"] / "adapter_config.json").exists()]
BY_ID = {m["id"]: m for m in MODELS}
MAX_TOKENS = 300   # bitta javob uchun eng ko'p token (modellar qisqa javobga o'qitilgan)
STARTED = int(time.time())

print("Model yuklanmoqda...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
STOP_IDS = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]   # o'qitilgan model javobni <|im_end|> bilan tugatadi
# GPU bo'lsa (Kaggle backend) float16 va GPU'da: javob ~2-3 soniya; CPU'da float32 (float16 CPU'da ishlamaydi).
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16 if DEVICE == "cuda" else torch.float32).to(DEVICE)
name = lambda model_id: model_id.replace(".", "_").replace("-", "_")      # noqa: E731  peft adapter nomi
model = None
for m in MODELS:
    if model is None:
        model = PeftModel.from_pretrained(base, ADAPTERS / m["id"], adapter_name=name(m["id"]))
    else:
        model.load_adapter(ADAPTERS / m["id"], adapter_name=name(m["id"]))
model.eval()
merged = None   # hozir base og'irliklarga qo'shilgan adapter
print("Tayyor:", ", ".join(BY_ID), "| qurilma:", DEVICE, flush=True)

# Bir vaqtda bitta generatsiya: boshqa so'rovlar navbatda kutadi. Ko'p foydalanuvchi uchun keyingi qadam — vLLM
# (continuous batching: bir nechta so'rov bitta GPU o'tishida), bu fayldagi API esa o'zgarmaydi.
gen_lock = threading.Lock()


def use(model_id):
    """Tanlangan adapterni base og'irliklarga qo'shadi (gen_lock ichida chaqiriladi)."""
    global merged
    if merged == model_id:
        return
    if merged is not None:
        model.unmerge_adapter()
    model.set_adapter(name(model_id))
    model.merge_adapter()
    merged = model_id


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


def start_generation(model_id, question, max_new_tokens, temperature=0.0, top_p=1.0):
    """Javobni alohida oqimda yozishni boshlaydi. job.streamer — matn bo'laklari, job.stop — to'xtatish,
    job.done — tugadi; tugagach job.prompt_tokens, job.completion_tokens, job.finish_reason to'ladi."""
    # Faqat bitta savol: modellar bitta savol -> bitta javob ko'rinishida o'qitilgan (chat_ui/server.py dagidek).
    prompt = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True)
    job = SimpleNamespace(streamer=TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True),
                          stop=threading.Event(), done=threading.Event(), prompt_tokens=0, completion_tokens=0,
                          finish_reason="stop", error=None)
    kwargs = dict(max_new_tokens=max(1, min(max_new_tokens, MAX_TOKENS)), repetition_penalty=1.1,
                  pad_token_id=tok.pad_token_id, eos_token_id=STOP_IDS, streamer=job.streamer,
                  stopping_criteria=StoppingCriteriaList([StopOnEvent(job.stop)]))
    # temperature=0 — greedy (har safar bir xil javob, test ham shunday o'tkazilgan); >0 — tasodifiy tanlash.
    kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p) if temperature > 0 else dict(do_sample=False))

    def run():
        with gen_lock:
            try:
                use(model_id)
                inp = tok(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**inp, **kwargs)
                job.prompt_tokens = inp.input_ids.shape[1]
                job.completion_tokens = out.shape[1] - inp.input_ids.shape[1]
                job.finish_reason = "stop" if out[0, -1].item() in STOP_IDS else "length"
            except Exception as e:
                print("Generatsiya xatosi:", repr(e), flush=True)
                job.error = repr(e)
                job.streamer.end()
            finally:
                job.done.set()

    threading.Thread(target=run, daemon=True).start()
    return job


app = FastAPI(title="TeachHelp API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.include_router(router)   # /auth/*, /keys, /usage — UI'dagi hisob va "API kalitlar" menyusi uchun


@app.exception_handler(ApiError)
def api_error(request, e):
    # OpenAI formatidagi xato: tayyor kutubxonalar uni o'qib, AuthenticationError / RateLimitError chiqaradi.
    kind = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error",
            404: "invalid_request_error", 429: "rate_limit_error", 503: "overloaded_error"}.get(e.status, "api_error")
    return JSONResponse({"error": {"message": e.message, "type": kind, "code": e.code}}, status_code=e.status)


@app.get("/")
def health():
    return {"ok": True, "models": list(BY_ID), "api": "/v1", "docs": "/docs", "ui": "/ui"}


# Chat sahifaning o'zi ham shu serverda (Vercel'siz, bitta tunnel havolasi yetadi): <server>/ui
CHAT_UI = HERE.parent.parent / "chat_ui"


@app.get("/ui")
def ui():
    return FileResponse(CHAT_UI / "index.html")


@app.get("/results.json")
def results():
    return FileResponse(CHAT_UI / "results.json")


@app.get("/config.js")
def config():
    return Response("window.API_BASE = location.origin;", media_type="text/javascript")   # API — shu server


# ---------- 1. UI: chat sahifa uchun (kalitsiz) ----------

class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = MODELS[0]["id"]
    messages: list[Message]
    max_new_tokens: int = 100


@app.get("/api/models")
def models():
    return MODELS


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in BY_ID or not req.messages:
        raise HTTPException(400, "Noto‘g‘ri so‘rov")
    job = start_generation(req.model, req.messages[-1].content, req.max_new_tokens)

    def stream():
        try:
            yield from job.streamer
        finally:
            job.stop.set()   # brauzer "to'xtatish"ni bossa, generatsiya ham to'xtaydi

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


# ---------- 2. API: dasturlar uchun, OpenAI formatida (API kalit bilan) ----------

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict] = ""   # OpenAI: matn yoki [{"type": "text", "text": "..."}]


class ChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None   # OpenAI'ning yangi nomi
    temperature: float = 0.0
    top_p: float = 1.0
    stream: bool = False
    stream_options: dict | None = None         # {"include_usage": true} — oxirgi bo'lakda token sarfi


def text_of(content):
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content if p.get("type") == "text")


@app.get("/v1/models")
def v1_models(authorization: str = Header(""), x_api_key: str = Header("")):
    check_api_key(authorization, x_api_key)
    return {"object": "list",
            "data": [{"id": m["id"], "object": "model", "created": STARTED, "owned_by": "teachhelp"} for m in MODELS]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest, authorization: str = Header(""), x_api_key: str = Header("")):
    key, remaining = check_api_key(authorization, x_api_key)
    if req.model not in BY_ID:
        raise ApiError(404, f"Model '{req.model}' topilmadi. Mavjud: {', '.join(BY_ID)}", "model_not_found")
    question = next((text_of(m.content) for m in reversed(req.messages) if m.role == "user"), "").strip()
    if not question:
        raise ApiError(400, "messages ichida 'user' xabari bo'lishi kerak", "invalid_request")
    job = start_generation(req.model, question, req.max_completion_tokens or req.max_tokens or MAX_TOKENS,
                           req.temperature, req.top_p)
    cid, created, t0 = "chatcmpl-" + secrets.token_hex(12), int(time.time()), time.time()
    headers = {"x-ratelimit-limit-requests": str(RATE_LIMIT), "x-ratelimit-remaining-requests": str(remaining)}

    def usage():
        return {"prompt_tokens": job.prompt_tokens, "completion_tokens": job.completion_tokens,
                "total_tokens": job.prompt_tokens + job.completion_tokens}

    def finish():
        """Generatsiya tugashini kutadi va sarfni yozadi (kalit egasi UI'da "Sarf" bo'limida ko'radi)."""
        job.done.wait(120)
        record_usage(key["id"], req.model, job.prompt_tokens, job.completion_tokens, int((time.time() - t0) * 1000))

    if not req.stream:
        text = "".join(job.streamer)
        finish()
        if job.error:
            raise ApiError(500, "Generatsiya xatosi: " + job.error, "server_error")
        return JSONResponse({
            "id": cid, "object": "chat.completion", "created": created, "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": job.finish_reason}],
            "usage": usage()}, headers=headers)

    def sse():
        # Server-Sent Events: har bo'lak "data: {...}\n\n", oxirida "data: [DONE]" — OpenAI stream formati.
        def chunk(delta, finish_reason=None):
            return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": req.model,
                                          "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]},
                                         ensure_ascii=False) + "\n\n"
        try:
            yield chunk({"role": "assistant", "content": ""})
            for piece in job.streamer:
                if piece:
                    yield chunk({"content": piece})
        finally:
            job.stop.set()   # mijoz ulanishni uzsa, generatsiya ham to'xtaydi
            finish()
        yield chunk({}, job.finish_reason)
        if (req.stream_options or {}).get("include_usage"):
            yield "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                         "model": req.model, "choices": [], "usage": usage()}) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers=headers)
