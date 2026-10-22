"""Hugging Face Space serveri: eng yaxshi modellar (Qwen2.5-1.5B + LoRA adapterlar), bepul CPU'da.

chat_ui/server.py bilan bir xil API (/api/models, /api/generate), shuning uchun chat_ui/index.html o'zgarishsiz
ishlaydi: Vercel'dagi sahifa so'rovlarni to'g'ridan-to'g'ri shu serverga yuboradi (config.js dagi API_BASE).
Base model Docker image yig'ilayotganda yuklab olinadi (Dockerfile), adapterlar esa Space repo'sining adapters/
papkasida (deploy/hf_space/prepare.py ularni float16'ga o'tkazib, u yerga ko'chiradi).

Xotira: bitta base model (1.5B, float32, ~6 GB) va bir nechta kichik adapter. Tanlangan adapter base og'irliklarga
qo'shib yuboriladi (merge): generatsiya oddiy model tezligida. Boshqa model tanlansa, eskisi ayirib olinadi.
"""
import os
import threading
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from peft import PeftModel
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

HERE = Path(__file__).parent
# Lokal sinov uchun BASE_PATH / ADAPTERS_PATH bilan loyihadagi papkalarni berish mumkin.
BASE = Path(os.environ.get("BASE_PATH", HERE / "base"))              # Dockerfile: snapshot_download("Qwen/Qwen2.5-1.5B")
ADAPTERS = Path(os.environ.get("ADAPTERS_PATH", HERE / "adapters"))  # prepare.py natijasi

# Faqat eng yaxshi modellar (bepul Space'da joy cheklangan). adapters/<id> bo'lmasa, model ro'yxatga chiqmaydi.
MODELS = [
    {"id": "sft-lora-dpo-v2", "label": "v3 · SFT + LoRA + DPO v2",
     "desc": "Yakuniy model, eng yaxshi natija · 44/68"},
    {"id": "sft-lora-rlhf-llm", "label": "v3 · SFT + LoRA + RLHF (Gemini hakam)",
     "desc": "Eng yaxshi RLHF: Gemini API baholagan, RLOO · 41/68"},
]
MODELS = [{**m, "base": "1.5b", "group": "Preference optimisation", "pretrained": False, "available": True}
          for m in MODELS if (ADAPTERS / m["id"] / "adapter_config.json").exists()]
BY_ID = {m["id"]: m for m in MODELS}

print("Model yuklanmoqda...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
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

gen_lock = threading.Lock()   # bir vaqtda bitta generatsiya: 2 ta CPU'da parallel so'rovlar faqat sekinlashtiradi


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


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = MODELS[0]["id"]
    messages: list[Message]
    max_new_tokens: int = 100


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"ok": True, "models": list(BY_ID)}


@app.get("/api/models")
def models():
    return MODELS


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in BY_ID or not req.messages:
        raise HTTPException(400, "Noto‘g‘ri so‘rov")
    # Faqat oxirgi savol: model bitta savol -> bitta javob ko'rinishida o'qitilgan (chat_ui/server.py dagidek).
    prompt = tok.apply_chat_template([req.messages[-1].model_dump()], tokenize=False, add_generation_prompt=True)

    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    stop = threading.Event()
    kwargs = dict(max_new_tokens=max(1, min(req.max_new_tokens, 300)), do_sample=False, repetition_penalty=1.1,
                  pad_token_id=tok.pad_token_id, eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")],
                  streamer=streamer, stopping_criteria=StoppingCriteriaList([StopOnEvent(stop)]))

    def run():
        with gen_lock:
            try:
                use(req.model)
                inp = tok(prompt, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    model.generate(**inp, **kwargs)
            except Exception as e:
                print("Generatsiya xatosi:", repr(e), flush=True)
                streamer.end()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        try:
            yield from streamer
        finally:
            stop.set()   # brauzer "to'xtatish"ni bossa, generatsiya ham to'xtaydi

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")
