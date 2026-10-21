"""RLHF v2, 1-qadam: reward model uchun ko'p juftlik -> rlhf/data_v2/train.jsonl, val.jsonl.

Ishga tushirish:  python rlhf/build_rm_data.py

RLHF v1'da reward model DPO v2 juftliklarida (339 ta, har promptdan bittadan) o'qitilgan edi va kuchsiz chiqdi:
qoida bali bilan korrelyatsiyasi +0.08, uzunlik bilan -0.63 ("qisqa — yaxshi"). Bu yerda o'sha 8 ta javobdan
(dpo/data_v2/generations.jsonl) ko'proq foyda olinadi:
  1. Har bir promptda 8 ta javob + namunaviy javob (SFT datadagi yoki qat'iy rad jumlasi) dpo/label.py qoidalari
     bilan baholanadi. Balli kamida MARGIN farq qiladigan HAR BIR juftlik olinadi (v1: faqat eng yaxshi va yomoni).
  2. Validation promptlar bo'yicha ajratiladi (v1: juftliklar bo'yicha, 18 ta): bir prompt ikkala tomonda bo'lmaydi.
  3. Uzunlik muvozanati har bir vazifada: chosen qisqaroq va uzunroq juftliklar soni teng bo'lguncha ko'pchilik
     tomon kamaytiriladi. Rad etishda bu qilinmaydi: u yerda to'g'ri javob (rad) tabiatan qisqa.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "dpo"))
import label  # noqa: E402

MARGIN = 0.5          # chosen balli rejected'nikidan kamida shuncha katta
MAX_PER_PROMPT = 8    # bitta prompt data'ni egallab olmasin
VAL_SHARE = 0.1       # promptlarning ulushi
REFUSALS = ("harmful", "cheating", "unknown")
rng = random.Random(42)


def golds(item):
    """Namunaviy javoblar: SFT datadagi javob; rad etishda yana qisqa qat'iy jumla. Rad etishda ikkalasi ham
    olinadi: SFT datadagisi uzunroq ("... It is dangerous and illegal."), reward model "rad = qisqa" deb o'rganmasin."""
    if item["task"] in ("harmful", "cheating"):
        return [label.FIXED[(item["task"], label.language(item["prompt"]) or "uz")], item["reference"]]
    return [item["reference"]]


def prompt_pairs(item):
    answers = {}   # bir xil matnli javoblar bitta deb olinadi
    for a in item["answers"]:
        if a["text"]:
            answers.setdefault(a["text"], a["finished"])
    for g in golds(item):
        answers.setdefault(g, True)
    scored = [(label.rule_score(item, t, f), t) for t, f in answers.items()]
    pairs = []
    for sc, c in scored:
        for sr, r in scored:
            if sc - sr < MARGIN:
                continue
            if item["task"] not in REFUSALS:
                ratio = len(c) / max(1, len(r))
                if not label.LENGTH_RATIO[0] <= ratio <= label.LENGTH_RATIO[1]:
                    continue
                if item["task"] not in ("arithmetic", "translate") and label.f1(c, r) >= 0.85:
                    continue   # deyarli bir xil javoblar (label.normal_pair'dagi kabi)
            pairs.append({"id": item["id"], "task": item["task"], "prompt": item["prompt"], "chosen": c,
                          "rejected": r, "margin": round(sc - sr, 3)})
    rng.shuffle(pairs)
    return pairs[:MAX_PER_PROMPT]


def balance(pairs):
    """Har bir vazifada chosen qisqaroq va uzunroq juftliklar soni tenglashtiriladi (rad etishdan tashqari)."""
    out = []
    for task in sorted({p["task"] for p in pairs}):
        mine = [p for p in pairs if p["task"] == task]
        if task in REFUSALS:
            out += mine
            continue
        short = [p for p in mine if len(p["chosen"]) < len(p["rejected"])]
        long_ = [p for p in mine if len(p["chosen"]) > len(p["rejected"])]
        same = [p for p in mine if len(p["chosen"]) == len(p["rejected"])]
        n = min(len(short), len(long_))
        out += rng.sample(short, n) + rng.sample(long_, n) + same
    rng.shuffle(out)
    return out


def main():
    items = [json.loads(line) for line in
             (ROOT / "dpo" / "data_v2" / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    ids = sorted(i["id"] for i in items)
    rng.shuffle(ids)
    val_ids = set(ids[:round(len(ids) * VAL_SHARE)])

    raw = [p for item in items for p in prompt_pairs(item)]
    split = {"train": balance([p for p in raw if p["id"] not in val_ids]),
             "val": balance([p for p in raw if p["id"] in val_ids])}
    out = HERE / "data_v2"
    out.mkdir(exist_ok=True)
    for name, rows in split.items():
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                           encoding="utf-8")

    rows = split["train"] + split["val"]
    print(f"Muvozanatdan oldin {len(raw)} juftlik, keyin {len(rows)} (train {len(split['train'])}, "
          f"val {len(split['val'])}, val promptlar {len(val_ids)}). v1: 339 + 18.")
    by_task = defaultdict(list)
    for p in rows:
        by_task[p["task"]].append(p)
    print(f"{'vazifa':<12} {'juftlik':>8} {'chosen qisqa':>13}")
    for task, mine in sorted(by_task.items()):
        print(f"{task:<12} {len(mine):>8} {sum(len(p['chosen']) < len(p['rejected']) for p in mine) / len(mine):>12.0%}")
    normal = [p for p in rows if p["task"] not in REFUSALS and len(p["chosen"]) != len(p["rejected"])]
    print(f"Rad etishsiz vazifalarda (uzunligi farq qiladigan juftliklar) chosen qisqaroq: "
          f"{sum(len(p['chosen']) < len(p['rejected']) for p in normal) / max(1, len(normal)):.0%} (v1 data: 56%)")
    print("Juftliklar soni promptlar bo'yicha:", dict(sorted(Counter(Counter(p["id"] for p in rows).values()).items())))


if __name__ == "__main__":
    main()
