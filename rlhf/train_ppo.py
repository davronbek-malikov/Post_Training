"""RLHF 3-qadam: SFT adapterni PPO bilan reward model bo'yicha o'qitish -> chat_ui/adapters/sft-lora-rlhf/.

Kaggle'da (GPU, T4 x2):  python rlhf/train_ppo.py
CPU'da sinov:  python rlhf/train_ppo.py --base Qwen2.5-0.5B --sft-adapter sft-lora --max-steps 1 --batch-size 2 --mini-batch 1 --max-new-tokens 16

Bu online o'qitish: har qadamda policy o'zi yangi javob yozadi, reward model baholaydi, policy yangilanadi.
Xotirada 4 ta model:
  policy    — base + SFT adapter, adapter o'qitiladi (GPU 0)
  reference — o'sha SFT model, muzlatilgan: KL penalty uchun (GPU 1)
  reward    — rlhf/models/<papka> (v1: 1_base-0.5b), muzlatilgan: javobga ball beradi (GPU 1)
  value     — reward model nusxasidan boshlanadi va o'qitiladi: har bir tokendan keyin kutiladigan reward (GPU 1)
Bitta qadam:
  1. Rollout: policy har bir promptga bitta javob yozadi.
  2. Reward: reward model balli (normallashtirilgan); to'xtamagan javobga qo'shimcha -1.
  3. KL penalty: har bir tokenda -kl_coef * (log policy - log reference). Policy SFT modeldan juda uzoqlashib,
     reward modelni aldab qo'ymasin (reward hacking).
  4. Advantage (GAE): har bir token value model kutganidan qancha yaxshi chiqdi.
  5. PPO update: ratio = policy_yangi / policy_eski, [1 - clip, 1 + clip] oralig'ida cheklanadi: bitta batch'dan
     juda katta qadam tashlanmaydi. Value model returns'ga yaqinlashtiriladi.

RLHF v2 (--algo rloo): value model o'rniga guruh. Har bir promptga --group-size ta javob yoziladi;
javobning advantage'i = uning reward'i - shu promptdagi boshqa javoblar reward'ining o'rtachasi (RLOO, 2024).
v1'da value model (vf_loss 1-7) har xil promptlarning har xil o'rtacha ballini taxmin qilishga urinardi va advantage
shovqinli edi; bir prompt javoblarini o'zaro solishtirish bu muammoni yo'qotadi va GPU'da bitta model kamayadi.
Clipped ratio loss va KL penalty o'sha-o'sha.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import judges
import scalar_model

HERE = Path(__file__).parent
ROOT = HERE.parent
ADAPTERS = ROOT / "chat_ui" / "adapters"
sys.path.insert(0, str(ROOT / "dpo"))
import label  # noqa: E402  DPO juftliklarini tuzgan qoidalar: bu yerda faqat tekshirish uchun, PPO ularni ko'rmaydi

# Reward modeldan mustaqil o'lchov. RM balli o'sib, bu ball tushsa — policy reward modelni aldayapti.
rule_score = label.rule_score


def load_policy(base, adapter, device):
    model = AutoModelForCausalLM.from_pretrained(ROOT / base, dtype=torch.float32).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    # is_trainable=True: SFT adapter muzlatilmaydi, PPO uni o'qitishni davom ettiradi (DPO'dagi kabi).
    model = PeftModel.from_pretrained(model, ADAPTERS / adapter, is_trainable=True)
    scalar_model.no_dropout(model)
    return model


def load_reference(base, adapter, device, dtype):
    model = AutoModelForCausalLM.from_pretrained(ROOT / base, dtype=dtype)
    model = PeftModel.from_pretrained(model, ADAPTERS / adapter).merge_and_unload()
    return model.to(device).eval().requires_grad_(False)


def response_part(x, prompts, responses, width):
    """x: [batch, uzunlik], i-pozitsiya (i+1)-token haqida. -> [batch, width]: faqat javob tokenlari, qolgani 0."""
    return torch.stack([F.pad(x[i, len(p) - 1:len(p) - 1 + len(r)], (0, width - len(r)))
                        for i, (p, r) in enumerate(zip(prompts, responses))])


def logprobs(model, prompts, responses, width, temperature, pad_id, device):
    """Javobning har bir tokeni uchun log p(token | oldingi matn). Sampling'dagi temperature bilan bir xil."""
    ids, mask = scalar_model.pad_batch([p + r for p, r in zip(prompts, responses)], pad_id, device)
    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].float() / temperature
    token_logp = -F.cross_entropy(logits.transpose(1, 2), ids[:, 1:], reduction="none")
    return response_part(token_logp, prompts, responses, width)


def values(model, prompts, responses, width, pad_id, device):
    """Value model: javobning har bir tokeni oldidagi holat uchun kutiladigan reward."""
    ids, mask = scalar_model.pad_batch([p + r for p, r in zip(prompts, responses)], pad_id, device)
    return response_part(model(ids, mask), prompts, responses, width)


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def gae(rewards, vals, gamma, lam):
    """Generalized Advantage Estimation. Padding joylarida reward va value 0, shuning uchun u yerda advantage ham 0."""
    adv, last = torch.zeros_like(rewards), 0.0
    for t in reversed(range(rewards.shape[1])):
        nxt = vals[:, t + 1] if t + 1 < rewards.shape[1] else 0.0
        delta = rewards[:, t] + gamma * nxt - vals[:, t]
        last = delta + gamma * lam * last
        adv[:, t] = last
    return adv, adv + vals   # advantages, returns (value model shu returns'ga o'rgatiladi)


@torch.no_grad()
def rollout(policy, tok, items, args, stop_ids, device):
    policy.eval()
    texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], tokenize=False,
                                     add_generation_prompt=True) for it in items]
    prompts = [tok(t)["input_ids"] for t in texts]
    inp = tok(texts, return_tensors="pt", padding=True).to(device)
    # Sof sampling (top_k/top_p/repetition_penalty yo'q): PPO javob ehtimolligini aynan shu taqsimotdan hisoblaydi.
    out = policy.generate(**inp, do_sample=True, temperature=args.temperature, top_k=0, top_p=1.0,
                          repetition_penalty=1.0, max_new_tokens=args.max_new_tokens, use_cache=True,
                          pad_token_id=tok.pad_token_id, eos_token_id=stop_ids)
    responses, finished = [], []
    for row in out[:, inp["input_ids"].shape[1]:].tolist():
        end = next((k for k, t in enumerate(row) if t in stop_ids), None)
        responses.append(row if end is None else row[:end + 1])   # to'xtash tokeni ham javobning bir qismi
        finished.append(end is not None)
    return prompts, responses, finished


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen2.5-1.5B")
    p.add_argument("--sft-adapter", default="sft-lora-v3", help="PPO boshlanadigan SFT adapter (policy va reference)")
    p.add_argument("--reward", default="1_base-0.5b", help="rlhf/models/ ichidagi reward model (v2: 2_sft-1.5b)")
    p.add_argument("--name", default="sft-lora-rlhf", help="natija adapter papkasi nomi")
    p.add_argument("--prompts", default="dpo/data/prompts.jsonl")
    p.add_argument("--epochs", type=int, default=1, help="promptlar ustidan necha marta o'tish")
    p.add_argument("--batch-size", type=int, default=8, help="bitta qadamdagi rollout (prompt) soni")
    p.add_argument("--mini-batch", type=int, default=4, help="PPO update va hisoblashlar shu bo'laklarda")
    p.add_argument("--ppo-epochs", type=int, default=2, help="bitta rollout batch ustida necha marta update")
    p.add_argument("--lr", type=float, default=5e-6, help="policy (DPO v2 bilan bir xil)")
    p.add_argument("--value-lr", type=float, default=5e-5)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--clip", type=float, default=0.2, help="policy ratio chegarasi")
    p.add_argument("--value-clip", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=150)
    p.add_argument("--no-stop-penalty", type=float, default=1.0, help="to'xtamagan javob balidan ayriladi")
    p.add_argument("--algo", choices=["ppo", "rloo"], default="ppo", help="ppo: value model (v1); rloo: guruh (v2)")
    p.add_argument("--group-size", type=int, default=4, help="rloo: bitta promptga nechta javob")
    p.add_argument("--reward-fp16", action="store_true", help="reward modelni float16'da yuklash (1.5B reward model)")
    p.add_argument("--judge", choices=["rm", "hf", "api"], default="rm",
                   help="rm: o'zimiz o'qitgan reward model (--reward); hf: tayyor HF reward model; api: katta LLM")
    p.add_argument("--judge-model", help="hf: model nomi (Skywork/...); api: model nomi (standart: JUDGE_MODEL)")
    p.add_argument("--rollouts", default="ppo_rollouts.jsonl", help="rlhf/logs/ ichida: har bir javob shu faylga")
    p.add_argument("--force", action="store_true", help="reward model uzunlik tekshiruvidan o'tmasa ham davom etish")
    p.add_argument("--limit-prompts", type=int, help="faqat birinchi N ta prompt (API hakamning kunlik limiti uchun)")
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    if args.algo == "rloo" and args.group_size < 2:
        raise SystemExit("rloo uchun --group-size kamida 2 bo'lishi kerak")

    random.seed(42)
    torch.manual_seed(42)
    cuda = torch.cuda.is_available()
    dev0 = "cuda:0" if cuda else "cpu"
    dev1 = "cuda:1" if cuda and torch.cuda.device_count() > 1 else dev0
    # Bitta GPU'da to'rt model 32 bitda sig'maydi: reference 16 bitda yuklanadi.
    ref_dtype = torch.float16 if cuda and dev1 == dev0 else torch.float32

    tok = AutoTokenizer.from_pretrained(ROOT / args.base)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"   # batch generatsiyada javob hamma qatorda bir joydan boshlanishi uchun
    pad = tok.pad_token_id
    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")]

    critic = args.algo == "ppo"
    own_rm = args.judge == "rm"
    if not own_rm and critic:
        raise SystemExit("--judge hf/api faqat --algo rloo bilan: value model o'zimizning reward modeldan boshlanadi")
    if own_rm:
        gate = json.loads((HERE / "models" / args.reward / "meta.json").read_text(encoding="utf-8")).get("length_gate")
        if gate and not gate["ok"] and not args.force:
            raise SystemExit(f"Reward model uzunlik tekshiruvidan o'tmagan: {gate}. U qisqa (yoki uzun) javobni "
                             "mukofotlaydi va PPO shuni o'rganadi (RLHF v1). Data'ni tuzating yoki --force.")
    print(f"Modellar yuklanmoqda: policy -> {dev0}; reference, reward{', value' if critic else ''} -> {dev1}", flush=True)
    policy = load_policy(args.base, args.sft_adapter, dev0)
    policy.print_trainable_parameters()
    reference = load_reference(args.base, args.sft_adapter, dev1, ref_dtype)
    length_coef = 0.0
    if own_rm:
        reward, meta = scalar_model.load(HERE / "models" / args.reward, dev1,
                                         dtype=torch.float16 if cuda and args.reward_fp16 else torch.float32)
        if critic:
            value, _ = scalar_model.load(HERE / "models" / args.reward, dev1, trainable=True)
            with torch.no_grad():   # value model normallashtirilgan ballar shkalasida boshlansin (reward kabi: o'rtacha 0)
                value.head.weight.div_(meta["std"])
                value.head.bias.sub_(meta["mean"]).div_(meta["std"])
        if len(AutoTokenizer.from_pretrained(ROOT / meta["base"])) != len(tok):
            raise SystemExit("Reward model va policy tokenizeri har xil: bir oiladagi modellar kerak (Qwen2.5).")
        print(f"Reward model: {meta['base']}, val acc {meta['val_acc']:.2f}; "
              f"ball = (r - {meta['mean']:+.2f}) / {meta['std']:.2f}")
        length_coef = meta.get("length_coef", 0.0)   # RLHF v2: length-controlled reward (train_reward.py)
        if length_coef:
            print(f"Uzunlik tuzatishi: r - {length_coef:+.4f} * belgilar soni")
    else:
        # Tayyor hakam: o'z shkalasi bor, shuning uchun ballar har qadamda normallashtiriladi (o'rtacha 0, og'ish 1).
        judge = (judges.HFReward(args.judge_model or "Skywork/Skywork-Reward-V2-Qwen3-1.7B", dev1)
                 if args.judge == "hf" else judges.LLMJudge(args.judge_model))
        print(f"Hakam: {args.judge} — {judge.name}", flush=True)

    opt_policy = torch.optim.AdamW([q for q in policy.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.0)
    if critic:
        opt_value = torch.optim.AdamW([q for q in value.parameters() if q.requires_grad], lr=args.value_lr,
                                      weight_decay=0.0)
    k = args.group_size if not critic else 1

    items = [json.loads(line) for line in (ROOT / args.prompts).read_text(encoding="utf-8").splitlines()]
    if args.limit_prompts:
        items = random.Random(0).sample(items, min(args.limit_prompts, len(items)))   # vazifalar aralash qolsin
    order = [it for _ in range(args.epochs) for it in random.sample(items, len(items))]
    batches = chunks(order, args.batch_size)[:args.max_steps]
    total = len(batches)

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    rollouts_file = (logs / args.rollouts).open("w", encoding="utf-8")
    log, start = [], time.time()
    print(f"{len(items)} prompt, {total} qadam. 1-qadamda KL ~0 bo'lishi kerak (policy = reference).", flush=True)
    for step, batch in enumerate(batches, 1):
        batch = [it for it in batch for _ in range(k)]   # rloo: har bir prompt k marta, ketma-ket (guruh = i // k)
        # ---------- 1. rollout ----------
        prompts, responses, finished = rollout(policy, tok, batch, args, stop_ids, dev0)
        width = max(len(r) for r in responses)
        mask = torch.tensor([[1.0] * len(r) + [0.0] * (width - len(r)) for r in responses])
        idx = chunks(list(range(len(batch))), args.mini_batch)

        def sub(seq, ids):
            return [seq[i] for i in ids]

        # ---------- 2-3. eski log-ehtimolliklar, reference, value, reward ----------
        with torch.no_grad():
            old = torch.cat([logprobs(policy, sub(prompts, b), sub(responses, b), width, args.temperature, pad, dev0).cpu()
                             for b in idx])
            ref = torch.cat([logprobs(reference, sub(prompts, b), sub(responses, b), width, args.temperature, pad, dev1)
                             .float().cpu() for b in idx])
            if critic:
                vals = torch.cat([values(value, sub(prompts, b), sub(responses, b), width, pad, dev1).cpu()
                                  for b in idx]) * mask
            if own_rm:
                raw = torch.cat([scalar_model.last_scores(reward, [prompts[i] + responses[i] for i in b], pad, dev1)
                                 .float().cpu() for b in idx])
        texts = [tok.decode(r, skip_special_tokens=True).strip() for r in responses]
        if own_rm:
            raw = raw - length_coef * torch.tensor([float(len(t)) for t in texts])   # v1'da length_coef = 0
            score = (raw - meta["mean"]) / meta["std"]
        else:
            raw = torch.tensor(judge.score(batch, texts, k) if args.judge == "api" else judge.score(batch, texts))
            score = (raw - raw.mean()) / (raw.std() + 1e-8)
        score = score - args.no_stop_penalty * (~torch.tensor(finished)).float()
        kl = (old - ref) * mask
        if critic:
            rewards = -args.kl_coef * kl
            for i, r in enumerate(responses):
                rewards[i, len(r) - 1] += score[i]   # reward model balli faqat oxirgi tokenda
            adv, returns = gae(rewards, vals, args.gamma, args.lam)
            mean = masked_mean(adv, mask)
            adv = (adv - mean) * torch.rsqrt(masked_mean((adv - mean) ** 2, mask) + 1e-8) * mask
        else:
            # Butun javob uchun bitta reward (KL penalty bilan), baseline = guruhdagi boshqa javoblarning o'rtachasi.
            seq_reward = (score - args.kl_coef * kl.sum(-1)).view(-1, k)
            seq_adv = (seq_reward - (seq_reward.sum(1, keepdim=True) - seq_reward) / (k - 1)).view(-1)
            seq_adv = seq_adv / (seq_adv.std() + 1e-8)
            adv = seq_adv[:, None] * mask   # javobning har bir tokeniga bir xil advantage

        # ---------- 5. PPO update ----------
        policy.train()
        stats = []
        for _ in range(args.ppo_epochs):
            for b in chunks(random.sample(range(len(batch)), len(batch)), args.mini_batch):
                m0, m1 = mask[b].to(dev0), mask[b].to(dev1)
                new = logprobs(policy, sub(prompts, b), sub(responses, b), width, args.temperature, pad, dev0)
                ratio = torch.exp(new - old[b].to(dev0))
                a = adv[b].to(dev0)
                pg_loss = masked_mean(torch.max(-a * ratio, -a * ratio.clamp(1 - args.clip, 1 + args.clip)), m0)
                pg_loss.backward()
                torch.nn.utils.clip_grad_norm_([q for q in policy.parameters() if q.requires_grad], 1.0)
                opt_policy.step()
                opt_policy.zero_grad()

                vf_loss = torch.zeros(())
                if critic:
                    v = values(value, sub(prompts, b), sub(responses, b), width, pad, dev1)
                    ov, ret = vals[b].to(dev1), returns[b].to(dev1)
                    v_clipped = ov + (v - ov).clamp(-args.value_clip, args.value_clip)
                    vf_loss = 0.5 * masked_mean(torch.max((v - ret) ** 2, (v_clipped - ret) ** 2), m1)
                    vf_loss.backward()
                    torch.nn.utils.clip_grad_norm_([q for q in value.parameters() if q.requires_grad], 1.0)
                    opt_value.step()
                    opt_value.zero_grad()

                with torch.no_grad():
                    d = new - old[b].to(dev0)
                    stats.append({"pg_loss": pg_loss.item(), "vf_loss": vf_loss.item(),
                                  "clipfrac": masked_mean(((ratio - 1).abs() > args.clip).float(), m0).item(),
                                  "approx_kl": 0.5 * masked_mean(d ** 2, m0).item()})

        # ---------- log ----------
        rules = [rule_score(it, t, f) for it, t, f in zip(batch, texts, finished)]
        for it, t, f, rm, rs in zip(batch, texts, finished, raw.tolist(), rules):
            rollouts_file.write(json.dumps({"step": step, "id": it["id"], "task": it["task"], "prompt": it["prompt"],
                                            "answer": t, "finished": f, "rm": round(rm, 3), "rule": round(rs, 3)},
                                           ensure_ascii=False) + "\n")
        rollouts_file.flush()
        s = {"step": step, "rm": raw.mean().item(), "rule": sum(rules) / len(rules),
             "kl": kl.sum(-1).mean().item(), "length": sum(map(len, responses)) / len(responses),
             "finished": sum(finished) / len(finished),
             **{k: sum(x[k] for x in stats) / len(stats) for k in stats[0]}}
        log.append({k: round(v, 4) for k, v in s.items()})
        elapsed = time.time() - start
        print(f"qadam {step}/{total} | RM ball {s['rm']:+.2f} | qoida bali {s['rule']:+.2f} | KL {s['kl']:.3f} | "
              f"uzunlik {s['length']:.0f} token | to'xtadi {s['finished']:.0%} | pg {s['pg_loss']:+.3f} | "
              f"vf {s['vf_loss']:.3f} | clip {s['clipfrac']:.2f} | qoldi ~{elapsed / step * (total - step) / 60:.0f} daq",
              flush=True)
        if step % 10 == 0 or step == total:
            print(f"  misol [{batch[0]['task']}] {batch[0]['prompt'][:80]}\n  -> {texts[0][:200]}", flush=True)
        if not args.max_steps and (step % 20 == 0 or step == total):
            policy.save_pretrained(ADAPTERS / args.name)   # Kaggle vaqti tugab qolsa ham oxirgi holat saqlansin
            (ADAPTERS / args.name / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")

    rollouts_file.close()
    if args.max_steps:
        print("Sinov tugadi, adapter saqlanmadi.")
        return
    first, last = log[:max(1, total // 10)], log[-max(1, total // 10):]
    for key in ("rm", "rule", "kl", "length"):
        print(f"{key:<7} boshida {sum(x[key] for x in first) / len(first):+.3f} -> oxirida {sum(x[key] for x in last) / len(last):+.3f}")
    print(f"Tayyor: {ADAPTERS / args.name} | javoblar: {logs / args.rollouts} | jami {(time.time() - start) / 60:.1f} daqiqa")


if __name__ == "__main__":
    main()
