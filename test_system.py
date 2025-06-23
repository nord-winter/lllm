#!/usr/bin/env python3
"""
Скрипт тестирования системы Qwen2.5-0.5B без fallback
Проверяет все компоненты и помогает диагностировать проблемы
"""

import sys
import traceback
import logging
import subprocess
import os
import time
import requests

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Тестирование импортов"""
    logger.info("🔍 Тестирование импортов...")
    
    tests = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("fastapi", "FastAPI"),
        ("gradio", "Gradio"),
        ("requests", "Requests")
    ]
    
    results = {}
    for module, name in tests:
        try:
            exec(f"import {module}")
            version = eval(f"{module}.__version__")
            results[name] = {"status": "✅", "version": version}
            logger.info(f"✅ {name}: {version}")
        except Exception as e:
            results[name] = {"status": "❌", "error": str(e)}
            logger.error(f"❌ {name}: {str(e)}")
    
    return results

def test_torch_setup():
    """Тестирование PyTorch"""
    logger.info("🔍 Тестирование PyTorch...")
    
    try:
        import torch
        logger.info(f"📦 PyTorch версия: {torch.__version__}")
        logger.info(f"🔧 CUDA доступна: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            logger.info(f"🎮 GPU устройств: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("💻 Будет использоваться CPU")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка PyTorch: {str(e)}")
        return False

def test_model_loading():
    """Тестирование загрузки модели Qwen2.5-0.5B без fallback"""
    logger.info("🔍 Тестирование загрузки модели Qwen2.5-0.5B...")
    
    try:
        from app.model import load_model
        
        logger.info("🤖 Загрузка модели (это может занять время при первом запуске)...")
        model, tokenizer = load_model()
        
        logger.info("✅ Модель загружена успешно!")
        logger.info(f"📊 Устройство: {model.device}")
        logger.info(f"🎯 Тип данных: {model.dtype}")
        
        # Простой тест генерации
        logger.info("🧪 Тестирование генерации...")
        test_input = "Привет! Как дела?"
        inputs = tokenizer(test_input, return_tensors="pt", padding=True)
        
        import torch
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"🤖 Тестовый ответ: {response}")
        logger.info("✅ Тест генерации прошел успешно")
        
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return False

def test_api_start():
    """Тестирование запуска API"""
    logger.info("🔍 Тестирование запуска API...")
    
    try:
        import uvicorn
        from app.main import app
        
        logger.info("✅ FastAPI приложение импортировано успешно")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка импорта API: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return False

def test_api_connection():
    """Тестирование подключения к запущенному API"""
    logger.info("🔍 Тестирование подключения к API...")
    
    api_url = "http://localhost:8000"
    
    try:
        # Проверяем health endpoint
        logger.info("🔍 Проверка health endpoint...")
        response = requests.get(f"{api_url}/health", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            logger.info("✅ API доступен!")
            logger.info(f"📊 Статус: {data.get('status')}")
            logger.info(f"🧠 Модель загружена: {data.get('model_loaded')}")
            
            # Тестируем генерацию через API
            logger.info("🧪 Тестирование генерации через API...")
            gen_response = requests.post(
                f"{api_url}/generate",
                json={
                    "text": "Привет! Расскажи интересный факт.",
                    "system_prompt": "Вы - полезный ассистент."
                },
                timeout=30
            )
            
            if gen_response.status_code == 200:
                result = gen_response.json()
                logger.info("✅ Генерация через API успешна!")
                logger.info(f"🤖 Ответ: {result['response'][:100]}...")
                return True
            else:
                logger.error(f"❌ Ошибка генерации: {gen_response.status_code}")
                return False
        else:
            logger.error(f"❌ API недоступен: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        logger.warning("⚠️ API недоступен (сервер не запущен)")
        logger.info("💡 Запустите API: uvicorn app.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка подключения к API: {str(e)}")
        return False

def test_ui_start():
    """Тестирование запуска UI"""
    logger.info("🔍 Тестирование запуска UI...")
    
    try:
        import gradio as gr
        from app.ui import demo
        
        logger.info("✅ Gradio UI импортирован успешно")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка импорта UI: {str(e)}")
        logger.error(f"📋 Трейсбек: {traceback.format_exc()}")
        return False

def check_huggingface_auth():
    """Проверка авторизации HuggingFace"""
    logger.info("🔍 Проверка авторизации HuggingFace...")
    
    try:
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            logger.info("✅ HuggingFace авторизация настроена")
            logger.info(f"👤 Пользователь: {result.stdout.strip()}")
            return True
        else:
            logger.warning("⚠️ HuggingFace авторизация не настроена")
            logger.info("💡 Запустите: huggingface-cli login")
            return False
    except FileNotFoundError:
        logger.warning("⚠️ huggingface-cli не найден")
        return False
    except Exception as e:
        logger.error(f"❌ Ошибка проверки HuggingFace: {str(e)}")
        return False

def show_recommendations(results):
    """Показать рекомендации по исправлению проблем"""
    logger.info("\n" + "="*50)
    logger.info("📋 РЕКОМЕНДАЦИИ ПО ИСПРАВЛЕНИЮ")
    logger.info("="*50)
    
    if not results.get('hf_auth', False):
        logger.info("🔧 Настройте авторизацию HuggingFace (рекомендуется):")
        logger.info("   huggingface-cli login")
    
    if not results.get('model_loading', False):
        logger.info("🔧 Если модель не загружается:")
        logger.info("   1. Проверьте подключение к интернету")
        logger.info("   2. Убедитесь в достаточном количестве памяти/диска")
        logger.info("   3. При первом запуске модель загружается из HuggingFace")
    
    if not results.get('api_connection', False):
        logger.info("🔧 Если API недоступен:")
        logger.info("   1. Запустите API сервер:")
        logger.info("      uvicorn app.main:app --host 0.0.0.0 --port 8000")
        logger.info("   2. Или используйте Docker:")
        logger.info("      docker-compose up --build")
    
    logger.info("\n🚀 Для запуска системы:")
    logger.info("   1. API: uvicorn app.main:app --host 0.0.0.0 --port 8000")
    logger.info("   2. UI: python app/ui.py")
    logger.info("   3. Или Docker: docker-compose up --build")

def main():
    """Основная функция тестирования"""
    logger.info("🚀 Запуск тестирования системы Qwen2.5-0.5B (без fallback)")
    logger.info("="*60)
    
    results = {}
    
    # Тестируем импорты
    import_results = test_imports()
    results['imports'] = import_results
    
    # Тестируем PyTorch
    results['torch'] = test_torch_setup()
    
    # Проверяем HuggingFace авторизацию
    results['hf_auth'] = check_huggingface_auth()
    
    # Тестируем компоненты приложения
    results['api'] = test_api_start()
    results['ui'] = test_ui_start()
    
    # Тестируем подключение к API (если запущен)
    results['api_connection'] = test_api_connection()
    
    # Тестируем загрузку модели (опционально)
    test_model = input("\n❓ Протестировать загрузку модели? (это может занять время) [y/N]: ").lower().strip()
    if test_model == 'y':
        results['model_loading'] = test_model_loading()
    else:
        logger.info("⏭️ Пропускаем тест загрузки модели")
        results['model_loading'] = None
    
    # Показываем итоговые результаты
    logger.info("\n" + "="*50)
    logger.info("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    logger.info("="*50)
    
    total_tests = 0
    passed_tests = 0
    
    for test_name, result in results.items():
        if test_name == 'imports':
            for lib_name, lib_result in result.items():
                total_tests += 1
                if lib_result['status'] == '✅':
                    passed_tests += 1
                    logger.info(f"✅ {lib_name}: {lib_result.get('version', 'OK')}")
                else:
                    logger.info(f"❌ {lib_name}: {lib_result.get('error', 'Ошибка')}")
        elif result is not None:
            total_tests += 1
            if result:
                passed_tests += 1
                logger.info(f"✅ {test_name}")
            else:
                logger.info(f"❌ {test_name}")
    
    logger.info(f"\n📈 Результат: {passed_tests}/{total_tests} тестов прошли успешно")
    
    if passed_tests == total_tests:
        logger.info("🎉 Все тесты прошли! Система готова к работе без fallback!")
    else:
        show_recommendations(results)

if __name__ == "__main__":
    main() 