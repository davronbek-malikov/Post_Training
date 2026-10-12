"""Lokal chat UI serveri: base Qwen2.5-0.5B va (keyinchalik) LoRA adapterlar.

Ishga tushirish:  python server.py   ->   http://127.0.0.1:8472
"""
import threading
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from transformers import (AutoModelForCausalLM, AutoTokenizer, StoppingCriteria,
                          StoppingCriteriaList, TextIteratorStreamer)

HERE = Path(__file__).parent
BASE_PATH = HERE.parent / "Qwen2.5-0.5B"
ADAPTERS = HERE / "adapters"   # adapters/<id>/adapter_config.json bo'lsa, model ro'yxatda yonadi

# Dropdown tartibi. Base'dan boshqa hammasi — base + adapters/<id> papkasidagi LoRA adapter.
MODELS = [
    {"id": "base",            "label": "Base (Pretrained)",   "desc": "Qwen2.5-0.5B — faqat keyingi so‘zni bashorat qiladi"},
    {"id": "sft-lora",        "label": "SFT + LoRA",          "desc": "LoRA bilan supervised fine-tuning"},
    {"id": "sft-qlora",       "label": "SFT + QLoRA",         "desc": "4-bit base ustida LoRA"},
    {"id": "sft-lora-rlhf",   "label": "SFT + LoRA + RLHF",   "desc": "LoRA SFT, so‘ng RLHF"},
    {"id": "sft-lora-dpo",    "label": "SFT + LoRA + DPO",    "desc": "LoRA SFT, so‘ng DPO"},
    {"id": "sft-qlora-rlhf",  "label": "SFT + QLoRA + RLHF",  "desc": "QLoRA SFT, so‘ng RLHF"},
    {"id": "sft-qlora-dpo",   "label": "SFT + QLoRA + DPO",   "desc": "QLoRA SFT, so‘ng DPO"},
]

cuda = torch.cuda.is_available()
dtype = torch.float16 if cuda else torch.float32   # CPU'da float16 ishlamaydi

print("Model yuklanmoqda:", BASE_PATH)
tok = AutoTokenizer.from_pretrained(BASE_PATH)
base = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=dtype).eval()
if cuda:
    base.cuda()
tok.pad_token = tok.pad_token or tok.eos_token
print("Qurilma:", base.device)

peft_model = None            # birinchi adapter so'ralganda yaratiladi
gen_lock = threading.Lock()  # bir vaqtda bitta generatsiya


def adapter_dir(model_id):
    return ADAPTERS / model_id


def is_available(model_id):
    return model_id == "base" or (adapter_dir(model_id) / "adapter_config.json").exists()


def ensure_adapter(model_id):
    global peft_model
    from peft import PeftModel
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(base, adapter_dir(model_id), adapter_name=model_id).eval()
    elif model_id not in peft_model.peft_config:
        peft_model.load_adapter(adapter_dir(model_id), adapter_name=model_id)
    peft_model.set_adapter(model_id)


class StopOnEvent(StoppingCriteria):
    def __init__(self, event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs):
        return self.event.is_set()


class Message(BaseModel):
    role: str
    content: str


class GenerateRequest(BaseModel):
    model: str = "base"
    messages: list[Message]
    max_new_tokens: int = 100


app = FastAPI()


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/models")
def models():
    return [{**m, "available": is_available(m["id"])} for m in MODELS]


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.model not in {m["id"] for m in MODELS} or not req.messages:
        raise HTTPException(400, "Noto‘g‘ri so‘rov")
    if not is_available(req.model):
        raise HTTPException(409, "Bu model hali o‘qitilmagan")

    if req.model == "base":
        # Base model chat formatini bilmaydi: 1-darsdagidek xom matn, faqat oxirgi xabar.
        prompt = req.messages[-1].content
    else:
        prompt = tok.apply_chat_template([m.model_dump() for m in req.messages],
                                         tokenize=False, add_generation_prompt=True)

    inp = tok(prompt, return_tensors="pt").to(base.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    stop = threading.Event()
    stop_ids = [tok.eos_token_id]
    if req.model != "base":
        stop_ids.append(tok.convert_tokens_to_ids("<|im_end|>"))   # o'qitilgan model javobni shu token bilan tugatadi
    kwargs = dict(**inp, max_new_tokens=max(1, min(req.max_new_tokens, 400)), do_sample=False,
                  repetition_penalty=1.1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids, streamer=streamer,
                  stopping_criteria=StoppingCriteriaList([StopOnEvent(stop)]))

    def run():
        with gen_lock:
            try:
                with torch.no_grad():
                    if req.model != "base":
                        ensure_adapter(req.model)
                        peft_model.generate(**kwargs)
                    elif peft_model is not None:
                        with peft_model.disable_adapter():
                            peft_model.generate(**kwargs)
                    else:
                        base.generate(**kwargs)
            except Exception as e:
                print("Generatsiya xatosi:", repr(e))
                streamer.end()

    threading.Thread(target=run, daemon=True).start()

    def stream():
        try:
            yield from streamer
        finally:
            stop.set()   # brauzer "to'xtatish"ni bossa, generatsiya ham to'xtaydi

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8472)
