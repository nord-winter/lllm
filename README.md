# LLM API и UI

Проект представляет собой API и веб-интерфейс для работы с языковыми моделями (LLM).

## Структура проекта

```
.
├── app/
│   ├── main.py    # FastAPI приложение
│   ├── model.py   # Модель и логика работы с LLM
│   └── ui.py      # Gradio веб-интерфейс
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Технологии

- FastAPI - для создания API
- Gradio - для веб-интерфейса
- Transformers - для работы с языковыми моделями
- PyTorch - для машинного обучения
- Docker - для контейнеризации

## Установка и запуск

1. Клонируйте репозиторий:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Запустите проект с помощью Docker Compose:
```bash
docker-compose up --build
```


После запуска:
- API будет доступен по адресу: http://localhost:8000
- Веб-интерфейс будет доступен по адресу: http://localhost:7860

## Разработка

Для локальной разработки:

1. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
.\venv\Scripts\activate  # для Windows
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

3. Запустите API:
```bash
uvicorn app.main:app --reload
```

4. В отдельном терминале запустите UI:
```bash
python app/ui.py
```