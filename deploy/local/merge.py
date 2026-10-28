"""1-qadam: DPO v2 LoRA adapterini asl Qwen2.5-1.5B ga qo'shib (merge), bitta to'liq model qilib saqlash.

Ishga tushirish:  python deploy/local/merge.py   ->   deploy/local/models/teachhelp-1.5b/
Model qayta o'qitilmaydi: W_yangi = W + (alpha / r) * B @ A — adapter matritsalari base og'irliklarga qo'shiladi xolos.
Natija oddiy Hugging Face model (adaptersiz), uni llama.cpp GGUF formatiga o'tkaza oladi (2-qadam).
"""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).parent.parent.parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="sft-lora-dpo-v2", help="chat_ui/adapters/ ichidagi papka (yakuniy model)")
    p.add_argument("--out", default=str(Path(__file__).parent / "models" / "teachhelp-1.5b"))
    args = p.parse_args()

    base = ROOT / "Qwen2.5-1.5B"
    print("Base:", base, "| adapter:", args.adapter)
    tok = AutoTokenizer.from_pretrained(base)
    # float32'da qo'shiladi (CPU'da aniqroq), keyin float16'da saqlanadi: fayl ~3 GB
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, ROOT / "chat_ui" / "adapters" / args.adapter).merge_and_unload()

    prompt = tok.apply_chat_template([{"role": "user", "content": "Python nima?"}], tokenize=False, add_generation_prompt=True)
    inp = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=60, do_sample=False, repetition_penalty=1.1, pad_token_id=tok.eos_token_id,
                             eos_token_id=[tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")])
    print("Sinov (merge'dan keyin):", tok.decode(out[0, inp.input_ids.shape[1]:], skip_special_tokens=True))

    model.to(torch.float16).save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print("Saqlandi:", args.out)


if __name__ == "__main__":
    main()
