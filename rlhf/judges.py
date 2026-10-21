"""O'zimiz o'qitmagan hakamlar (reward): RL paytida javobga ball beradi. train_ppo.py --judge hf | api.

  HFReward  — Hugging Face'dagi tayyor reward model (masalan, Skywork-Reward-V2). O'z tokenizeri bor:
              policy javobi matnga qaytariladi va qaytadan tokenlanadi (Skywork Qwen3, policy esa Qwen2.5).
  LLMJudge  — katta LLM API orqali (Groq, OpenAI, OpenRouter, Gemini — OpenAI'ga mos API'si borlar).
              Bitta so'rov bitta promptning BARCHA javoblarini birga baholaydi: RLOO uchun shu kerak (javoblar
              o'zaro solishtiriladi) va so'rovlar soni 4 barobar kam. Javoblar tartibi aralashtiriladi: LLM hakamlar
              birinchi ko'rgan javobni yaxshiroq ko'rishga moyil (position bias).

Sozlash (LLMJudge): JUDGE_API_KEY (majburiy), JUDGE_BASE_URL (standart: Groq), JUDGE_MODEL.
  Kaggle'da: Add-ons -> Secrets. Lokal kompyuterda: loyiha ildizidagi .env fayli (git'ga tushmaydi, .gitignore).
"""
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

GROQ = "https://api.groq.com/openai/v1"


def load_env(path=Path(__file__).parent.parent / ".env"):
    """.env dagi KALIT=qiymat qatorlari; muhitda allaqachon bor kalitlar o'zgartirilmaydi."""
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.strip().partition("=")
            if sep and key and not key.startswith("#"):
                os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


class HFReward:
    def __init__(self, name, device):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        self.tok.padding_side = "right"
        self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
        dtype = torch.float16 if str(device).startswith("cuda") else torch.bfloat16
        self.model = AutoModelForSequenceClassification.from_pretrained(name, dtype=dtype, num_labels=1)
        self.model.config.pad_token_id = self.tok.pad_token_id   # oxirgi (pad bo'lmagan) token qaerdaligini topish uchun
        self.model = self.model.to(device).eval()
        self.device, self.name = device, name

    def text(self, prompt, answer):
        # Model kartasidagi kabi: chat template, boshidagi takroriy bos token olib tashlanadi.
        t = self.tok.apply_chat_template([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}],
                                         tokenize=False)
        return t[len(self.tok.bos_token):] if self.tok.bos_token and t.startswith(self.tok.bos_token) else t

    @torch.no_grad()
    def score(self, items, answers, batch=8):
        out = []
        for i in range(0, len(answers), batch):
            texts = [self.text(it["prompt"], a) for it, a in zip(items[i:i + batch], answers[i:i + batch])]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            out += self.model(**enc).logits[:, 0].float().tolist()
        return out


INSTRUCTIONS = """You grade answers written by a small tutoring assistant. Questions are in Uzbek or English.
Score every answer from 1 (bad) to 10 (excellent) on:
- correctness (numbers, facts, translation meaning);
- answering in the language of the question (translation tasks: in the target language);
- following the instruction exactly (e.g. "3 ta" / "3 items" means exactly 3 numbered items, one per line);
- harmful requests or cheating during an exam: a short polite refusal is the best answer;
- questions nobody can know (the future, private facts): "Bilmayman" / "I don't know" is the best answer;
- broken or invented words, repetition, or an answer that does not stop are bad.
Do NOT reward length: a short correct answer beats a long one with the same content.
A reference answer is given to help you; other correct answers are fine too.
Reply with JSON only: {"scores": [s1, s2, ...]} in the same order as the answers."""


class LLMJudge:
    def __init__(self, model=None, base_url=None, api_key=None, workers=8, retries=6):
        load_env()
        self.model = model or os.environ.get("JUDGE_MODEL", "openai/gpt-oss-120b")
        self.url = (base_url or os.environ.get("JUDGE_BASE_URL", GROQ)).rstrip("/") + "/chat/completions"
        self.key = api_key or os.environ.get("JUDGE_API_KEY")
        if not self.key:
            raise SystemExit("JUDGE_API_KEY topilmadi (Kaggle'da: Add-ons -> Secrets).")
        self.workers, self.retries, self.rng = workers, retries, random.Random(0)
        self.calls = self.failed = 0
        self.name = self.model

    def ask(self, item, answers):
        """Bitta prompt, bir nechta javob -> ballar ro'yxati (javoblar tartibida). Xato bo'lsa None."""
        import requests
        order = list(range(len(answers)))
        self.rng.shuffle(order)
        shown = "\n\n".join(f"Answer {n + 1}:\n{answers[j] or '(empty)'}" for n, j in enumerate(order))
        content = f"Question:\n{item['prompt']}\n\nReference answer:\n{item.get('reference', '-')}\n\n{shown}"
        body = {"model": self.model, "temperature": 0,
                "messages": [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": content}]}
        last = "limit (429) yoki server xatosi"
        for attempt in range(self.retries):
            try:
                r = requests.post(self.url, headers={"Authorization": f"Bearer {self.key}"}, json=body, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:   # limit yoki server: kutib qayta urinish
                    time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                    continue
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                found = re.search(r"\{.*\}", text, re.S)
                shown_scores = [float(s) for s in json.loads(found.group(0))["scores"]]
                if len(shown_scores) != len(answers):
                    raise ValueError(f"{len(shown_scores)} ta ball, {len(answers)} ta javob")
                scores = [0.0] * len(answers)
                for n, j in enumerate(order):
                    scores[j] = shown_scores[n]
                return scores
            except (requests.RequestException, ValueError, KeyError, AttributeError, TypeError) as e:
                last = e
                time.sleep(2 ** attempt)
        print(f"  hakam xatosi ({item['prompt'][:40]}...): {last!r}", flush=True)
        return None

    def score(self, items, answers, k):
        """items/answers: har bir prompt k marta ketma-ket (train_ppo.py dagi kabi). Guruhlar parallel so'raladi."""
        groups = [(items[i], answers[i:i + k]) for i in range(0, len(answers), k)]
        with ThreadPoolExecutor(self.workers) as pool:
            results = list(pool.map(lambda g: self.ask(*g), groups))
        out = []
        for res in results:
            self.calls += 1
            if res is None:
                self.failed += 1
                res = [5.0] * k   # hammasi bir xil: RLOO'da advantage 0, bu promptdan hech narsa o'rganilmaydi
            out += res
        return out
