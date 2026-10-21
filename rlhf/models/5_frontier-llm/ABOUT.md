# 5. Reward model: katta LLM, API orqali (standart: Groq, openai/gpt-oss-120b)

| | |
|---|---|
| Reward model | Katta LLM RL paytida har bir javobni jonli baholaydi (1-10). OpenAI'ga mos API (Groq, OpenAI, OpenRouter) |
| O'qitilganmi? | Reward model — **yo'q**. SFT model — **ha** (RLOO bilan) |
| So'rovlar | bitta so'rov = bitta promptning 4 ta javobi (tartibi aralash); bepul tarif uchun 400 prompt |
| Kalit | Kaggle: Add-ons -> Secrets -> `JUDGE_API_KEY` |
| Kod | `rlhf/judges.py` (`LLMJudge`), `train_ppo.py --judge api`, notebook: shu papkadagi `rlhf_llm_judge_kaggle.ipynb` |

Model o'qitilmagani uchun bu papkada og'irliklar yo'q: hakam provayderning serverida ishlaydi.
