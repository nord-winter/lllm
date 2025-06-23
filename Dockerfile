FROM python:3.10

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install huggingface_hub[hf_xet]

# Увеличиваем таймаут загрузки моделей
ENV HF_HUB_DOWNLOAD_TIMEOUT=600

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]