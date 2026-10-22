# 5. Reward model: katta LLM, API orqali (standart: Google Gemini, gemini-3.5-flash-lite)

| | |
|---|---|
| Reward model | Katta LLM RL paytida har bir javobni jonli baholaydi (1-10). OpenAI'ga mos API (Groq, OpenAI, OpenRouter) |
| O'qitilganmi? | Reward model — **yo'q**. SFT model — **ha** (RLOO bilan) |
| So'rovlar | bitta so'rov = bitta promptning 4 ta javobi (tartibi aralash); bepul tarif uchun 400 prompt |
| Kalit | https://aistudio.google.com/apikey; Kaggle: Add-ons -> Secrets -> `JUDGE_API_KEY`; lokal: `.env` |
| 40 val juftlikda | Gemini 3.5 Flash-Lite **0.90** (teng ball 0); Groq gpt-oss-120b 0.82 (teng 5); bizning v2 RM 0.84 |
| Bepul limit | `gemini-3.5-flash` — kuniga 20 so'rov (yetmaydi); Flash-Lite ishladi; Groq'ning token limiti kichik |
| Kod | `rlhf/judges.py` (`LLMJudge`), `train_ppo.py --judge api`, notebook: shu papkadagi `rlhf_llm_judge_kaggle.ipynb` |

Model o'qitilmagani uchun bu papkada og'irliklar yo'q: hakam provayderning serverida ishlaydi.
