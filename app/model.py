from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model():
    """
    Загружает только модель Qwen2.5-0.5B-Instruct, без fallback
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    
    try:
        logger.info(f"🔄 Загрузка модели: {model_name}")
        
        # Загружаем токенизатор
        logger.info("⚙️ Загрузка токенизатора...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Загружаем модель
        logger.info("🧠 Загрузка модели...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            trust_remote_code=True
        )
        
        logger.info(f"✅ Модель {model_name} успешно загружена")
        logger.info(f"📊 Устройство модели: {model.device}")
        logger.info(f"🎯 Тип данных: {model.dtype}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка загрузки модели {model_name}: {str(e)}")
        logger.error("🚨 Система не может работать без основной модели")
        raise Exception(f"Не удалось загрузить модель {model_name}: {str(e)}")

def test_model_loading():
    """
    Тестовая функция для проверки загрузки модели
    """
    try:
        logger.info("🧪 Запуск тестирования загрузки модели...")
        model, tokenizer = load_model()
        
        # Простой тест генерации
        logger.info("🧪 Тестирование генерации...")
        test_input = "Привет! Как дела?"
        inputs = tokenizer(test_input, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🧪 Тестовый ответ: {response}")
        logger.info("✅ Тест загрузки модели прошел успешно")
        return True
    except Exception as e:
        logger.error(f"❌ Тест загрузки модели провален: {str(e)}")
        return False