# 2. Backend: model (engine) — llama-server

Engine — modelni xotiraga yuklab, javob yozadigan dastur. Biz **llama.cpp** ning `llama-server.exe` dasturini
ishlatamiz: CPU'da tez ishlaydi va OpenAI formatidagi API beradi. Kod yozmaganmiz, faqat sozlaganmiz.
GPU server olinsa, uning o'rniga **vLLM** qo'yiladi (gateway o'zgarmaydi, faqat `ENGINE_URL`).

## Model qanday tayyorlandi

| Qadam | Nima | Natija |
|---|---|---|
| 1 | [merge.py](merge.py): LoRA adapter + Qwen2.5-1.5B -> bitta model | `hybrid/models/teachhelp-1.5b/` (3 GB, Hugging Face formati) |
| 2 | llama.cpp `convert_hf_to_gguf.py --outtype q8_0` | `hybrid/models/teachhelp-1.5b-q8_0.gguf` (1.65 GB) |
| 3 | [eval_api.py](eval_api.py): 68 savol bilan tekshirish | kvantlash modelni buzmaganini ko'rsatadi |
| — | Qwen3.5-4B: tayyor GGUF yuklab olindi | `hybrid/models/Qwen3.5-4B-Q4_K_M.gguf` (2.74 GB) |

**GGUF** — llama.cpp (ggml) ning model formati: og'irliklar, tokenizer va chat shabloni bitta faylda.
**q8_0** — har bir og'irlik 8 bit: hajm 2 baravar kichik, sifat deyarli o'zgarmaydi.

## llama-server sozlamalari ([5_runner/run.py](../5_runner/run.py) ichida)

| Parametr | Bizda | Ma'nosi |
|---|---|---|
| `-m` | `models/...gguf` | model fayli |
| `--port` | 8080 / 8081 | har bir model o'z portida |
| `--parallel` | 2 | bir vaqtda nechta so'rov |
| `--ctx-size` | 2048 / 8192 | kontekst uzunligi (barcha parallel so'rovlarga) |
| `--threads` | CPU yadrolari − 2 | nechta CPU oqimi ishlatiladi |
| `--jinja` | yoqilgan | modelning o'z chat shabloni ishlatiladi |
| `--reasoning-budget 0` | faqat Qwen | "thinking" o'chiq: CPU'da javob tezroq |

Har bir model uchun generatsiya sozlamalari (repeat penalty, max tokens) esa
[1_backend/gateway.py](../1_backend/gateway.py) dagi `MODELS` ro'yxatida.
