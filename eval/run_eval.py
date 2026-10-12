"""Evaluation suite: eval_set.json savollarini modelga beradi va javoblarni saqlaydi.

Ishga tushirish:  python run_eval.py                   ->  results/base.json
                  python run_eval.py --model sft-lora  ->  results/sft-lora.json
                  python run_eval.py --limit 5         ->  faqat birinchi 5 ta savol
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
BASE_PATH = HERE.parent / "Qwen2.5-0.5B"
ADAPTERS = HERE.parent / "chat_ui" / "adapters"   # chat UI bilan bir xil papka

APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʻʼ`"})


def norm(text):
    return text.translate(APOSTROPHES).lower()


def auto_check(item, answer):
    """True/False — avtomatik tekshiruv natijasi. Qolgan mezonlarni (til, toza tugash) odam baholaydi."""
    if "items" in item:   # "3 ta" -> roppa-rosa 3 ta band
        return len(re.findall(r"^\s*(?:\d+[.)]|[-•*])\s+\S", answer, flags=re.M)) == item["items"]
    # Savolni qaytargan javob o'tmasin: savol matni javobdan olib tashlanadi.
    a = norm(answer).replace(norm(item["messages"][0]["content"]), " ")
    for key in map(norm, item["accept"]):
        # Son alohida turishi kerak: "8" javobi "18" yoki "0:08" ichidan topilmasin.
        if re.search(rf"(?<![\d:]){re.escape(key)}(?![\d:])", a) if key.isdigit() else key in a:
            return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="base")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=100)
    args = p.parse_args()

    items = json.loads((HERE / "eval_set.json").read_text(encoding="utf-8"))[:args.limit]

    cuda = torch.cuda.is_available()
    tok = AutoTokenizer.from_pretrained(BASE_PATH)
    model = AutoModelForCausalLM.from_pretrained(BASE_PATH, dtype=torch.float16 if cuda else torch.float32).eval()
    if cuda:
        model.cuda()
    tok.pad_token = tok.pad_token or tok.eos_token
    stop_ids = [tok.eos_token_id]
    if args.model != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTERS / args.model).eval()
        stop_ids.append(tok.convert_tokens_to_ids("<|im_end|>"))   # o'qitilgan model javobni shu token bilan tugatadi

    results = []
    for i, item in enumerate(items, 1):
        user = item["messages"][0]
        if args.model == "base":
            # Base model chat formatini bilmaydi: xom matn (chat UI'dagi kabi).
            prompt = user["content"]
        else:
            prompt = tok.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        inp = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.1, pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
        answer = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        results.append({
            "id": item["id"],
            "task": item["task"],
            "messages": [user, {"role": "assistant", "content": answer}],
            "reference": item["messages"][1]["content"],
            "auto_pass": auto_check(item, answer),
        })
        print(f"[{i}/{len(items)}] {item['id']}: {'OK' if results[-1]['auto_pass'] else '--'}")

    out_path = HERE / "results" / f"{args.model}.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    total, passed = Counter(r["task"] for r in results), Counter(r["task"] for r in results if r["auto_pass"])
    print(f"\n{args.model}: avtomatik tekshiruv")
    for task in total:
        print(f"  {task:<11} {passed[task]}/{total[task]}")
    print(f"  {'jami':<11} {sum(passed.values())}/{len(results)}")
    print("Saqlandi:", out_path)


if __name__ == "__main__":
    main()
