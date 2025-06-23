from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model import load_model, test_model_loading
from fastapi.responses import HTMLResponse, JSONResponse
from typing import Optional
import traceback
import torch
import logging
import sys

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM API", description="API для работы с Qwen2.5-0.5B")

# Ленивая загрузка модели
model = None
tokenizer = None
model_info = {}

def get_model():
    global model, tokenizer, model_info
    if model is None or tokenizer is None:
        try:
            logger.info("🔄 Инициализация модели...")
            model, tokenizer = load_model()
            
            # Сохраняем информацию о модели
            model_info = {
                "model_class": str(type(model).__name__),
                "tokenizer_class": str(type(tokenizer).__name__),
                "device": str(model.device) if hasattr(model, 'device') else "unknown",
                "dtype": str(model.dtype) if hasattr(model, 'dtype') else "unknown",
                "model_name": getattr(model, "name_or_path", "unknown"),
                "vocab_size": tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else "unknown"
            }
            logger.info(f"✅ Модель загружена: {model_info}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
            logger.error(f"📋 Полный трейсбек: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели: {str(e)}")
    return model, tokenizer

class Prompt(BaseModel):
    text: str
    system_prompt: Optional[str] = "Вы - полезный ассистент, способный отвечать на различные вопросы."

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head>
            <title>LLM API - Диагностика</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .success { background-color: #d4edda; color: #155724; }
                .error { background-color: #f8d7da; color: #721c24; }
                .info { background-color: #d1ecf1; color: #0c5460; }
            </style>
        </head>
        <body>
            <h1> LLM API - Qwen2.5-0.5B</h1>
            <div class="info status">
                <h3>📊 Диагностические эндпоинты:</h3>
                <ul>
                    <li><a href="/health">/health</a> - Проверка здоровья API</li>
                    <li><a href="/model-info">/model-info</a> - Информация о модели</li>
                    <li><a href="/test-model">/test-model</a> - Тест загрузки модели</li>
                    <li><a href="/docs">/docs</a> - OpenAPI документация</li>
                </ul>
            </div>
            
            <div class="info status">
                <h3>🚀 API эндпоинты:</h3>
                <ul>
                    <li><code>POST /generate</code> - Генерация текста</li>
                    <li><code>POST /simple-chat</code> - Простой тестовый чат</li>
                </ul>
            </div>
            
            <p>🌐 UI доступен по адресу <a href="http://localhost:7860" target="_blank">http://localhost:7860</a></p>
        </body>
    </html>
    """

@app.get("/health")
def health():
    try:
        model, tokenizer = get_model()
        return {
            "status": "healthy", 
            "model_loaded": True,
            "model_info": model_info,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        logger.error(f"❌ Health check failed: {str(e)}")
        return {
            "status": "unhealthy", 
            "model_loaded": False, 
            "error": str(e),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }

@app.get("/model-info")
def model_info_endpoint():
    """Подробная информация о модели"""
    try:
        model, tokenizer = get_model()
        return {
            "model_info": model_info,
            "transformers_version": __import__('transformers').__version__,
            "torch_version": torch.__version__,
            "device_info": {
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None
            }
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/test-model")
def test_model_endpoint():
    """Тестирование загрузки модели"""
    try:
        success = test_model_loading()
        return {"test_passed": success, "message": "Тест загрузки модели завершен"}
    except Exception as e:
        return {"test_passed": False, "error": str(e), "traceback": traceback.format_exc()}

@app.post("/generate")
def generate(prompt: Prompt):
    """Генерация текста из текстового промпта"""
    try:
        logger.info(f"📝 Получен запрос на генерацию: {prompt.text[:50]}...")
        model, tokenizer = get_model()
        
        # Создаем простой промпт для генерации
        input_text = f"System: {prompt.system_prompt}\nUser: {prompt.text}\nAssistant:"
        logger.info(f"🔤 Промпт создан, длина: {len(input_text)}")
        
        # Обрабатываем входы
        logger.info("⚙️ Обработка входных данных...")
        
        # Проверяем тип токенизатора
        if hasattr(tokenizer, 'tokenizer'):
            # Это AutoTokenizer
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        else:
            # Это AutoTokenizer
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=2048)
        
        inputs = inputs.to(model.device)
        logger.info(f"📊 Входы подготовлены, устройство: {model.device}")
        
        # Генерируем ответ
        logger.info("🧠 Генерация ответа...")
        with torch.no_grad():
            # Определяем pad_token_id для генерации
            pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            eos_token_id = tokenizer.eos_token_id
            
            output_ids = model.generate(
                **inputs, 
                max_new_tokens=512, 
                do_sample=True, 
                temperature=0.7,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Декодируем результат
        logger.info("🔤 Декодирование результата...")
        full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Извлекаем только новый сгенерированный текст
        if "Assistant:" in full_output:
            generated_text = full_output.split("Assistant:")[-1].strip()
        else:
            generated_text = full_output[len(input_text):].strip()
        
        logger.info(f"✅ Генерация завершена, длина ответа: {len(generated_text)}")
        return {"response": generated_text, "input_length": len(input_text), "output_length": len(generated_text)}
        
    except Exception as e:
        logger.error(f"❌ Ошибка генерации: {str(e)}")
        logger.error(f"📋 Полный трейсбек: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")

@app.post("/simple-chat")
def simple_chat(prompt: Prompt):
    """Упрощенный чат для тестирования"""
    try:
        logger.info(f"💬 Простой чат: {prompt.text}")
        return {
            "response": f"Простой тестовый ответ на: '{prompt.text}'. API работает! 🎉", 
            "status": "test_mode",
            "timestamp": __import__('datetime').datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Ошибка в простом чате: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")

# Добавляем startup event для предварительной загрузки модели
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Запуск API...")
    logger.info("ℹ️ Модель будет загружена при первом запросе")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Завершение работы API...")