"""Deploy qilingan modelni o'sha 68 ta test savoli bilan baholash — OpenAI formatidagi istalgan API orqali.

Ishga tushirish:
  python deploy/local/eval_api.py --url https://<server>/v1 --key sk-th-... --name local-gguf-q8
  python deploy/local/eval_api.py --url http://127.0.0.1:8080/v1 --name gguf-q8      # llama-server'ning o'zi
Natija eval/results/<name>.json ga yoziladi (eval/run_eval.py bilan bir xil format), eval/summarize.py jadvalga qo'shadi.
Kaggle'dagi DPO v2 natijasi 44/68: GGUF (8-bit) nusxa shunga yaqin bo'lsa, kvantlash modelni buzmagan.
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from openai import OpenAI

EVAL = Path(__file__).parent.parent.parent / "eval"
sys.path.insert(0, str(EVAL))
from run_eval import auto_check  # noqa: E402  avtomatik tekshiruv run_eval.py bilan bir xil


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True, help="API base URL, /v1 bilan tugaydi")
    p.add_argument("--key", default="none", help="sk-th-... (gateway uchun); llama-server'ga kalit kerak emas")
    p.add_argument("--model", default="sft-lora-dpo-v2")
    p.add_argument("--name", required=True, help="natija fayli nomi: eval/results/<name>.json")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-new-tokens", type=int, default=100)
    args = p.parse_args()

    client = OpenAI(base_url=args.url, api_key=args.key, max_retries=5)
    items = json.loads((EVAL / "eval_set.json").read_text(encoding="utf-8"))[:args.limit]
    results, t0 = [], time.time()
    for i, item in enumerate(items, 1):
        user = item["messages"][0]
        # run_eval.py dagidek: greedy, repetition penalty 1.1 (llama-server qo'shimcha maydonlarni tushunadi)
        r = client.chat.completions.create(model=args.model, messages=[user], max_tokens=args.max_new_tokens, temperature=0,
                                           extra_body={"repeat_penalty": 1.1, "repeat_last_n": 1024})
        answer = (r.choices[0].message.content or "").strip()
        results.append({"id": item["id"], "task": item["task"], "messages": [user, {"role": "assistant", "content": answer}],
                        "reference": item["messages"][1]["content"], "auto_pass": auto_check(item, answer)})
        print(f"[{i}/{len(items)}] {item['id']}: {'OK' if results[-1]['auto_pass'] else '--'}", flush=True)

    out_path = EVAL / "results" / f"{args.name}.json"
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    total, passed = Counter(r["task"] for r in results), Counter(r["task"] for r in results if r["auto_pass"])
    print(f"\n{args.name}: avtomatik tekshiruv ({(time.time() - t0) / len(results):.1f} s/savol)")
    for task in total:
        print(f"  {task:<11} {passed[task]}/{total[task]}")
    print(f"  {'jami':<11} {sum(passed.values())}/{len(results)}")
    print("Saqlandi:", out_path)


if __name__ == "__main__":
    main()
