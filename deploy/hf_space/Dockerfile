FROM python:3.12-slim

# Hugging Face Spaces konteynerni 1000-foydalanuvchi sifatida ishga tushiradi
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Base model image ichiga yuklanadi: Space qayta ishga tushganda qayta yuklab olinmaydi
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B', local_dir='base')"

COPY --chown=user app.py .
COPY --chown=user adapters adapters

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
