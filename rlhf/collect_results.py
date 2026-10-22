"""Har bir reward-model tajribasining natijalarini o'z papkasiga yig'adi: rlhf/models/<papka>/results/ va RESULTS.md.

Ishga tushirish:  python rlhf/collect_results.py      (yangi tajriba tugagandan keyin)

Nusxa olinadigan fayllar (asl fayllar o'z joyida qoladi: eval/summarize.py va chat UI ulardan foydalanadi):
  eval/results/<model>.json                 -> results/test_answers.json   (68 ta test savoliga javoblar)
  chat_ui/adapters/<model>/training_log.json -> results/rl_training_log.json (RL'ning har qadami)
  rlhf/logs/<rollouts>.jsonl                -> results/rollouts.jsonl     (RL paytida yozilgan har bir javob va bali)
"""
import json
import shutil
import statistics as st
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent
TASKS = [("factual", "Fakt"), ("explain", "Tushuntirish"), ("list", "Ro'yxat"), ("translate", "Tarjima"),
         ("arithmetic", "Arifmetika"), ("unknown", "\"Bilmayman\""), ("harmful", "Zararli so'rovni rad etish"),
         ("cheating", "Imtihonda ko'chirishni rad etish")]

EXPERIMENTS = [
    {"folder": "1_base-0.5b", "model": "sft-lora-rlhf", "rollouts": "ppo_rollouts.jsonl", "k": 1,
     "title": "Base model (Qwen2.5-0.5B) reward model + PPO — RLHF v1"},
    {"folder": "2_sft-1.5b", "model": "sft-lora-rlhf-v2", "rollouts": "rlhf_v2_rollouts.jsonl", "k": 4,
     "title": "SFT modelimizning nusxasi (Qwen2.5-1.5B + SFT v3) reward model + RLOO — RLHF v2"},
    {"folder": "3_open-llm-7b", "model": None, "title": "Kattaroq ochiq LLM (7B) — hali qilinmagan"},
    {"folder": "4_skywork-1.7b", "model": "sft-lora-rlhf-skywork", "rollouts": "rlhf_skywork_rollouts.jsonl", "k": 4,
     "title": "Tayyor reward model (Skywork-Reward-V2-Qwen3-1.7B) + RLOO"},
    {"folder": "5_frontier-llm", "model": "sft-lora-rlhf-llm", "rollouts": "rlhf_llm_rollouts.jsonl", "k": 4,
     "title": "Katta LLM hakam (Gemini 3.5 Flash-Lite, API) + RLOO"},
]


def corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else 0.0


def scores(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for r in rows:
        p, t = out.get(r["task"], (0, 0))
        out[r["task"]] = (p + bool(r["auto_pass"]), t + 1)
    chars = round(st.mean(len(r["messages"][-1]["content"]) for r in rows))
    return out, chars


def main():
    sft, _ = scores(ROOT / "eval" / "results" / "sft-lora-v3.json")
    dpo, _ = scores(ROOT / "eval" / "results" / "sft-lora-dpo-v2.json")
    total = lambda s: f"{sum(p for p, _ in s.values())}/{sum(t for _, t in s.values())}"  # noqa: E731
    for exp in EXPERIMENTS:
        folder = HERE / "models" / exp["folder"]
        lines = [f"# Natijalar: {exp['title']}", ""]
        if not exp["model"]:
            lines += ["Bu tajriba hali o'tkazilmagan: natija yo'q."]
            (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"{exp['folder']}: natija yo'q")
            continue
        out = folder / "results"
        out.mkdir(exist_ok=True)
        files = {"test_answers.json": ROOT / "eval" / "results" / f"{exp['model']}.json",
                 "rl_training_log.json": ROOT / "chat_ui" / "adapters" / exp["model"] / "training_log.json",
                 "rollouts.jsonl": ROOT / "rlhf" / "logs" / exp["rollouts"]}
        for name, src in files.items():
            shutil.copy(src, out / name)

        mine, chars = scores(files["test_answers.json"])
        lines += ["Test: 68 ta savol (`eval/eval_set.json`), avtomatik tekshiruv. Hamma tajriba SFT v3 dan boshlangan.", "",
                  "| Vazifa | SFT v3 (boshlanish) | **Bu tajriba** | DPO v2 (eng yaxshi) |", "|---|---|---|---|"]
        for task, name in TASKS:
            lines.append(f"| {name} | {sft[task][0]}/{sft[task][1]} | **{mine[task][0]}/{mine[task][1]}** | "
                         f"{dpo[task][0]}/{dpo[task][1]} |")
        lines.append(f"| **Jami** | **{total(sft)}** | **{total(mine)}** | **{total(dpo)}** |")

        log = json.loads(files["rl_training_log.json"].read_text(encoding="utf-8"))
        n = max(1, len(log) // 10)
        avg = lambda part, key: st.mean(x[key] for x in part)  # noqa: E731
        rows = [json.loads(line) for line in files["rollouts.jsonl"].read_text(encoding="utf-8").splitlines()]
        lines += ["", "## RL jarayoni", "", "| Ko'rsatkich | Boshida (10% qadam) | Oxirida (10% qadam) |", "|---|---|---|"]
        for key, name in (("rm", "Hakam bali"), ("rule", "Qoida bali (hakamdan mustaqil tekshiruv)"),
                          ("kl", "KL (SFT'dan uzoqlashish)"), ("length", "Javob uzunligi (token)")):
            lines.append(f"| {name} | {avg(log[:n], key):.2f} | {avg(log[-n:], key):.2f} |")
        lines += ["", f"- Qadamlar: {len(log)}; RL paytida yozilgan javoblar: {len(rows)}",
                  f"- Hakam bali va qoida bali korrelyatsiyasi: {corr([r['rm'] for r in rows], [r['rule'] for r in rows]):+.2f}",
                  f"- Hakam bali va javob uzunligi korrelyatsiyasi: "
                  f"{corr([r['rm'] for r in rows], [len(r['answer']) for r in rows]):+.2f}",
                  f"- Test javoblarining o'rtacha uzunligi: {chars} belgi"]
        if exp["k"] > 1:
            agree = pairs = 0
            for i in range(0, len(rows), exp["k"]):
                g = rows[i:i + exp["k"]]
                if len({r["rm"] for r in g}) == 1:
                    continue   # hamma ball bir xil = hakam javob bermagan (API limit): o'lchovga kirmaydi
                for a in range(len(g)):
                    for b in range(a + 1, len(g)):
                        if g[a]["rule"] != g[b]["rule"] and g[a]["answer"] != g[b]["answer"]:
                            pairs += 1
                            agree += (g[a]["rm"] > g[b]["rm"]) == (g[a]["rule"] > g[b]["rule"])
            lines.append(f"- Bir savolning 4 ta javobi ichida hakam qoidalar bilan mos: {agree / max(1, pairs):.0%} "
                         f"({pairs} juftlik). RLOO aynan shu farqlardan o'rganadi")
        lines += ["", "## Fayllar (`results/`)", "",
                  "| Fayl | Nima |", "|---|---|",
                  "| `test_answers.json` | 68 ta test savoli, model javobi, to'g'ri javob va `auto_pass` |",
                  "| `rl_training_log.json` | RL'ning har qadami: hakam bali, qoida bali, KL, uzunlik, loss |",
                  "| `rollouts.jsonl` | RL paytida yozilgan har bir javob, uning hakam bali (`rm`) va qoida bali (`rule`) |"]
        (folder / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{exp['folder']}: test {total(mine)}, {len(log)} qadam -> {out}")


if __name__ == "__main__":
    main()
