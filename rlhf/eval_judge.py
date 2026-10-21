"""Tayyor hakam bizning juftliklarimizda qanchalik to'g'ri? RL'dan oldingi tekshiruv (o'qitish yo'q).

Ishga tushirish:
  python rlhf/eval_judge.py --judge hf --model Skywork/Skywork-Reward-V2-Qwen3-1.7B   # CPU'da ham ishlaydi
  python rlhf/eval_judge.py --judge api --model openai/gpt-oss-120b                   # JUDGE_API_KEY kerak

O'zimizning v2 reward model (Qwen2.5-1.5B + SFT v3) xuddi shu 181 ta val juftlikda: aniqlik 0.84, uzunlik
tekshiruvi pair_corr -0.35 (tuzatishsiz). Tayyor hakam bundan yaxshi bo'lmasa, uni RL'da ishlatishdan foyda yo'q.
Tayyor modellar asosan ingliz tilidagi data'da o'qitilgan, shuning uchun o'zbekcha savollar alohida ko'rsatiladi.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dpo"))
import judges  # noqa: E402
import label  # noqa: E402
from train_reward import length_gate  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", choices=["hf", "api"], default="hf")
    p.add_argument("--model", help="hf: Skywork/...; api: model nomi (standart: JUDGE_MODEL)")
    p.add_argument("--data", default="rlhf/data_v2/val.jsonl")
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    pairs = [json.loads(line) for line in (ROOT / args.data).read_text(encoding="utf-8").splitlines()][:args.limit]
    # Val juftliklarida namunaviy javob yo'q: LLM hakamga yordam uchun generations.jsonl dan olinadi.
    refs = {json.loads(line)["id"]: json.loads(line)["reference"] for line in
            (ROOT / "dpo" / "data_v2" / "generations.jsonl").read_text(encoding="utf-8").splitlines()}
    items = [{**pair, "reference": refs.get(pair["id"], "-")} for pair in pairs for _ in range(2)]
    answers = [a for pair in pairs for a in (pair["chosen"], pair["rejected"])]

    start = time.time()
    if args.judge == "hf":
        judge = judges.HFReward(args.model or "Skywork/Skywork-Reward-V2-Qwen3-1.7B",
                                "cuda" if torch.cuda.is_available() else "cpu")
        scores = judge.score(items, answers, batch=4)
    else:
        judge = judges.LLMJudge(args.model)
        scores = judge.score(items, answers, k=2)   # bitta so'rov = bitta juftlik (2 ta javob, tartibi aralash)
        print(f"API so'rovlar: {judge.calls}, xato: {judge.failed}")
    if args.judge == "api" and judge.last_failed:
        # Javob olinmagan juftliklar (429 limit va h.k.) hakamning xatosi emas: hisobdan chiqariladi.
        keep = [i for i in range(len(pairs)) if i not in set(judge.last_failed)]
        print(f"Javob olinmagan {len(judge.last_failed)} juftlik hisobga olinmaydi")
        pairs, scores = [pairs[i] for i in keep], [x for i in keep for x in scores[2 * i:2 * i + 2]]
    sc, sr = torch.tensor(scores[0::2]), torch.tensor(scores[1::2])
    right = (sc > sr).tolist()   # teng ball (LLM'da bo'lishi mumkin) — xato deb hisoblanadi
    print(f"{len(pairs)} juftlik, {time.time() - start:.0f} s")

    print(f"\n{judge.name}: val acc {sum(right) / len(right):.2f}. Bizning v2 RM: 0.84")
    for name, lang in (("o'zbekcha savollar", "uz"), ("inglizcha savollar", "en")):
        mine = [ok for pair, ok in zip(pairs, right) if label.language(pair["prompt"]) == lang]
        if mine:
            print(f"  {name}: {sum(mine) / len(mine):.2f} ({len(mine)} juftlik)")
    for task in sorted({pair["task"] for pair in pairs}):
        mine = [ok for pair, ok in zip(pairs, right) if pair["task"] == task]
        print(f"  [{task}]: {sum(mine) / len(mine):.2f} ({len(mine)} juftlik)")
    if args.judge == "api":
        print(f"  teng ball: {sum(float(c) == float(r) for c, r in zip(sc, sr))} juftlikda")
    print("Uzunlik tekshiruvi (rad etishsiz):", length_gate(pairs, sc, sr), "| bizning v2 RM: pair_corr -0.35")


if __name__ == "__main__":
    main()
