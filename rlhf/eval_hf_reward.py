"""Tayyor reward model (Hugging Face) bizning juftliklarimizda qanchalik yaxshi hakam? RL'dan oldingi tekshiruv.

Ishga tushirish:  python rlhf/eval_hf_reward.py --model Skywork/Skywork-Reward-V2-Qwen3-1.7B
CPU'da ham ishlaydi (bfloat16, 1.7B uchun ~4 GB xotira, ~10 daqiqa).

O'zimizning v2 reward model (Qwen2.5-1.5B + SFT v3) xuddi shu 181 ta val juftlikda: aniqlik 0.84, uzunlik
tekshiruvi pair_corr -0.35 (tuzatishsiz). Tayyor model bundan yaxshi bo'lmasa, uni RL'da ishlatishdan foyda yo'q.
Tayyor modellar asosan ingliz tilidagi juftliklarda o'qitilgan, shuning uchun o'zbekcha savollarni alohida ko'ramiz.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dpo"))
import label  # noqa: E402
from train_reward import length_gate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Skywork/Skywork-Reward-V2-Qwen3-1.7B")
    p.add_argument("--data", default="rlhf/data_v2/val.jsonl")
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, dtype=torch.float16 if device == "cuda" else torch.bfloat16, num_labels=1).to(device).eval()
    pairs = [json.loads(line) for line in (ROOT / args.data).read_text(encoding="utf-8").splitlines()][:args.limit]

    @torch.no_grad()
    def score(prompt, answer):
        # Model kartasidagi kabi: suhbat chat template bilan, boshidagi takroriy bos token olib tashlanadi.
        text = tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                                       tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]
        return model(**tok(text, return_tensors="pt").to(device)).logits[0][0].float().item()

    start, sc, sr = time.time(), [], []
    for i, pair in enumerate(pairs, 1):
        sc.append(score(pair["prompt"], pair["chosen"]))
        sr.append(score(pair["prompt"], pair["rejected"]))
        if i % 20 == 0 or i == len(pairs):
            print(f"{i}/{len(pairs)} juftlik | {time.time() - start:.0f} s", flush=True)
    sc, sr = torch.tensor(sc), torch.tensor(sr)
    right = (sc > sr).tolist()

    print(f"\n{args.model}: val acc {sum(right) / len(right):.2f} ({len(right)} juftlik). Bizning v2 RM: 0.84")
    for name, keep in (("o'zbekcha savollar", lambda q: label.language(q) == "uz"),
                       ("inglizcha savollar", lambda q: label.language(q) == "en")):
        mine = [ok for pair, ok in zip(pairs, right) if keep(pair["prompt"])]
        if mine:
            print(f"  {name}: {sum(mine) / len(mine):.2f} ({len(mine)} juftlik)")
    for task in sorted({pair["task"] for pair in pairs}):
        mine = [ok for pair, ok in zip(pairs, right) if pair["task"] == task]
        print(f"  [{task}]: {sum(mine) / len(mine):.2f} ({len(mine)} juftlik)")
    print("Uzunlik tekshiruvi (rad etishsiz):", length_gate(pairs, sc, sr), "| bizning v2 RM: pair_corr -0.35")


if __name__ == "__main__":
    main()
