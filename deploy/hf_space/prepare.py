"""Space uchun adapterlarni tayyorlash: chat_ui/adapters/<id> -> deploy/hf_space/adapters/<id>, float16'da.

Ishga tushirish:  python deploy/hf_space/prepare.py
Adapterlar Kaggle'da float32'da saqlangan (~280 MB). Float16'da yarmi (~140 MB) va javoblar farq qilmaydi:
app.py ularni yuklaganda base model kabi float32'ga qaytaradi. Faqat app.py dagi MODELS ro'yxatidagilar olinadi.
"""
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).parent
SOURCE = HERE.parent.parent / "chat_ui" / "adapters"
DEPLOY = ["sft-lora-dpo-v2", "sft-lora-rlhf-llm"]   # app.py dagi MODELS bilan bir xil


def main():
    out = HERE / "adapters"
    shutil.rmtree(out, ignore_errors=True)
    for model_id in DEPLOY:
        src = SOURCE / model_id
        if not (src / "adapter_config.json").exists():
            print(f"{model_id}: topilmadi, o'tkazib yuborildi")
            continue
        (out / model_id).mkdir(parents=True)
        shutil.copy(src / "adapter_config.json", out / model_id)
        weights = {k: v.to(torch.float16) for k, v in load_file(src / "adapter_model.safetensors").items()}
        save_file(weights, out / model_id / "adapter_model.safetensors")
        size = (out / model_id / "adapter_model.safetensors").stat().st_size / 2 ** 20
        print(f"{model_id}: {size:.0f} MB")


if __name__ == "__main__":
    main()
