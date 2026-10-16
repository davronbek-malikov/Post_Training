"""DPO 3-qadam: javoblarni baholab, chosen/rejected juftliklarini tuzish -> dpo/data/train.jsonl, val.jsonl.

Ishga tushirish:  python dpo/label.py

Har bir javobga qoidalar asosida ball qo'yiladi:
  - tekshirsa bo'ladigan vazifalar (rad etish, "Bilmayman", arifmetika, ro'yxat): 1 — to'g'ri, 0 — noto'g'ri;
  - qolganlari (tushuntirish, faktlar, tarjima...): SFT datadagi to'g'ri javobga so'zlar bo'yicha o'xshashlik (0..1);
  - jarimalar: keraksiz "Kechirasiz", to'xtamaslik, takrorlanish, haddan tashqari uzunlik, boshqa tilda javob.
Eng yuqori ballli javob — chosen, eng pasti — rejected. Hamma javob yomon bo'lsa, chosen — SFT datadagi javob.
Farq aniq bo'lmasa, prompt tashlab yuboriladi: g'olibi yo'q juftlik modelga hech narsa o'rgatmaydi.
"""
import ast
import json
import random
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
VAL_SHARE = 0.05
rng = random.Random(42)

REFUSAL_OK = re.compile(r"yordam bera olmayman|can't help|cannot help", re.I)
REFUSAL_ANY = re.compile(r"kechirasiz|olmayman|\bsorry\b|can't help", re.I)
UNKNOWN_OK = re.compile(r"^(bilmayman|i don't know)\b", re.I)
UZ_MARK = re.compile(r"[‘’ʻʼ]|\b(va|bu|uchun|bilan|nima|emas|ham|bo'l|qil|bor)\w*", re.I)
CHECKABLE = {"harmful", "cheating", "unknown", "arithmetic", "list", "translate"}
APOSTROPHES = str.maketrans({c: "'" for c in "‘’ʻʼ`"})


def words(text):
    return re.findall(r"[\w']+", text.translate(APOSTROPHES).lower())


def f1(a, b):
    """So'zlar bo'yicha o'xshashlik: 1 — bir xil so'zlar, 0 — umumiy so'z yo'q."""
    wa, wb = Counter(words(a)), Counter(words(b))
    common = sum((wa & wb).values())
    if not common:
        return 0.0
    p, r = common / sum(wa.values()), common / sum(wb.values())
    return 2 * p * r / (p + r)


OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
       ast.Div: lambda a, b: a // b}


def calc(expr):
    """Faqat butun sonlar va + - * / ni hisoblaydi (eval o'rniga: model matnidagi "9**9**9" kabi narsa ishlamaydi)."""
    def walk(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            return OPS[type(n.op)](walk(n.left), walk(n.right))
        raise ValueError("ruxsat etilmagan ifoda")
    return walk(ast.parse(expr.strip(), mode="eval").body)


def arithmetic_ok(text, number):
    """Birinchi natija to'g'ri va matndagi har bir "a op b = c" tenglik ham to'g'ri bo'lishi kerak."""
    first = re.search(r"=\s*(\d+)", text)
    if not first or int(first.group(1)) != number:
        return False
    for lhs, rhs in re.findall(r"([\d\s+\-*/()]*\d[\d\s+\-*/()]*)=\s*(\d+)", text):
        try:
            if calc(lhs) != int(rhs):
                return False
        except (SyntaxError, ValueError, ZeroDivisionError):
            pass   # "Avval ... 4 / 2" kabi qismlar to'liq ifoda emas: tekshirilmaydi
    return True


def task_score(item, text):
    task = item["task"]
    if task in ("harmful", "cheating"):
        return 1.0 if REFUSAL_OK.search(text) and len(text) < 250 else 0.0
    if task == "unknown":
        if not UNKNOWN_OK.match(text) or REFUSAL_ANY.search(text):
            return 0.0
        # "Bilmayman." dan keyin ma'nosiz gap qo'shilgan bo'lsa, faqat yarim ball: chosen bo'la olmaydi.
        return 1.0 if len(words(text)) <= 2 or f1(text, item["reference"]) >= 0.5 else 0.5
    if task == "translate":
        # So'zlar o'xshashligi yetmaydi: "hot" o'rniga "cold" deyilsa ham o'xshashlik 0.8 chiqadi.
        # To'g'ri tarjimaning hamma so'zi bo'lsa — 1, aks holda ko'pi bilan 0.6.
        ref, ans = set(words(item["reference"])), words(text)
        if ref <= set(ans) and len(ans) <= 1.5 * len(ref) + 1:
            return 1.0
        return 0.6 * f1(text, item["reference"])
    if task == "arithmetic":
        return 1.0 if arithmetic_ok(text, item["number"]) else 0.0
    if task == "list":
        n = len(re.findall(r"^\s*\d+[.)]\s+\S", text, flags=re.M))
        return 1.0 if n == item["items"] and not REFUSAL_ANY.search(text) else 0.0
    return f1(text, item["reference"])


def penalty(item, answer):
    text, task, why = answer["text"], item["task"], []
    if task not in ("harmful", "cheating") and REFUSAL_ANY.search(text):
        why.append("keraksiz rad")
    if not answer["finished"]:
        why.append("to'xtamadi")
    sentences = [s.strip() for s in re.split(r"[.\n]", text) if len(s.strip()) > 10]
    if len(sentences) != len(set(sentences)):
        why.append("takror")
    if len(text) > 300 and len(text) > 3 * len(item["reference"]):
        why.append("juda uzun")
    if task != "translate":
        q_uz = bool(UZ_MARK.search(item["prompt"]))
        if q_uz and len(words(text)) >= 6 and not UZ_MARK.search(text):
            why.append("inglizcha javob")
        if not q_uz and re.search(r"[‘’ʻʼ]", text):
            why.append("o'zbekcha javob")
    return why


def score(item, answer):
    if not answer["text"]:
        return -1.0, ["bo'sh"]
    why = penalty(item, answer)
    return task_score(item, answer["text"]) - 0.5 * len(why), why


def make_pair(item):
    """(chosen, rejected, manba) yoki None."""
    scored = sorted(((score(item, a), a["text"]) for a in item["answers"]), key=lambda x: (x[0][0], -len(x[1])))
    (worst, worst_why), rejected = scored[0]
    (best, _), chosen = scored[-1]
    good = 1.0 if item["task"] in CHECKABLE else 0.5    # "yaxshi javob" chegarasi
    if best >= good and best - worst >= 0.5 and chosen != rejected:
        return chosen, rejected, "model", worst_why
    if best < good / 2 and worst < good / 2:
        # Model bu savolga umuman yaxshi javob bermadi: chosen — SFT datadagi to'g'ri javob.
        return item["reference"], rejected, "reference", worst_why
    return None


def main():
    items = [json.loads(line) for line in (HERE / "data" / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    pairs, skipped = [], Counter()
    for item in items:
        pair = make_pair(item)
        if pair is None:
            skipped[item["task"]] += 1
            continue
        chosen, rejected, source, why = pair
        pairs.append({"id": item["id"], "task": item["task"], "source": source, "rejected_why": why,
                      "prompt": item["prompt"], "chosen": chosen, "rejected": rejected})

    rng.shuffle(pairs)
    n_val = max(1, round(len(pairs) * VAL_SHARE))
    for name, rows in (("val.jsonl", pairs[:n_val]), ("train.jsonl", pairs[n_val:])):
        (HERE / "data" / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8")

    print(f"{'vazifa':<12} {'juftlik':>8} {'model':>6} {'ref':>5} {'tashlandi':>10}")
    for task in sorted({i["task"] for i in items}):
        mine = [p for p in pairs if p["task"] == task]
        print(f"{task:<12} {len(mine):>8} {sum(p['source'] == 'model' for p in mine):>6} "
              f"{sum(p['source'] == 'reference' for p in mine):>5} {skipped[task]:>10}")
    print(f"{'jami':<12} {len(pairs):>8}  (train {len(pairs) - n_val}, val {n_val})")
    why = Counter(w for p in pairs for w in p["rejected_why"])
    print("Rejected javoblardagi jarimalar:", dict(why.most_common()))

    print("\nTekshirish uchun misollar (har vazifadan bittadan):")
    for task in sorted({p["task"] for p in pairs}):
        p = next(p for p in pairs if p["task"] == task)
        print(f"\n[{task}, {p['source']}] {p['prompt']}\n  + {p['chosen'][:150]}\n  - {p['rejected'][:150]}")


if __name__ == "__main__":
    main()
