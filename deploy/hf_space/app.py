"""Hugging Face Space serveri: faqat DPO v2 modeli (Qwen2.5-1.5B + sft-lora-dpo-v2), CPU'da.

chat_ui/server.py bilan bir xil API (/api/models, /api/generate), shuning uchun chat_ui/index.html o'zgarishsiz
ishlaydi: Vercel /api/* so'rovlarini shu serverga yo'naltiradi (chat_ui/vercel.json).
Base model Docker image yig'ilayotganda yuklab olinadi (Dockerfile), adapter esa Space repo'sining adapter/ papkasida.
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
# Lokal sinov uchun BASE_PATH / ADAPTER_PATH bilan loyihadagi papkalarni berish mumkin.
BASE = Path(os.environ.get("BASE_PATH", HERE / "base"))           # Dockerfile: snapshot_download("Qwen/Qwen2.5-1.5B")
ADAPTER = Path(os.environ.get("ADAPTER_PATH", HERE / "adapter"))  # chat_ui/adapters/sft-lora-dpo-v2 nusxasi

MODEL = {"id": "sft-lora-dpo-v2", "base": "1.5b", "group": "Preference optimisation",
         "label": "v3 · SFT + LoRA + DPO v2", "desc": "Toza rad javoblari, uzunlik muvozanati, SFT qismi · 44/68",
         "pretrained": False, "available": True}

print("Model yuklanmoqda...", flush=True)
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)   # CPU'da float16 ishlamaydi
# Adapter base og'irliklarga qo'shib yuboriladi: bitta model qoladi, generatsiya tezroq.
model = PeftModel.from_pretrained(model, ADAPTER).merge_and_unload().eval()
print("Tayyor", flush=True)

gen_lock = threading.Lock()   # bir vaqtda bitta generatsiya: 2 ta CPU'da parallel so'rovlar faqat sekinlashtiradi


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = MODEL["id"]
    messages: list[Message]
    max_new_tokens: int = 100


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
def health():
    return {"ok": True, "model": MODEL["id"]}


@app.get("/api/models")
def models():
    return [MODEL]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model != MODEL["id"] or not req.messages:
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
                inp = tok(prompt, return_tensors="pt")
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
