import gradio as gr
import requests
import time
import os
import traceback
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API URL
API_URL = "http://api:8000" if os.environ.get("DOCKER_ENV") else "http://localhost:8000"

def get_api_status():
    """Получение детального статуса API"""
    try:
        # Проверяем health
        health_response = requests.get(f"{API_URL}/health", timeout=10)
        health_data = health_response.json()
        
        # Проверяем model-info
        model_response = requests.get(f"{API_URL}/model-info", timeout=10)
        model_data = model_response.json()
        
        status_text = "✅ **API Статус: Работает**\n\n"
        status_text += f"**Health Status:** {health_data.get('status', 'unknown')}\n"
        status_text += f"**Модель загружена:** {health_data.get('model_loaded', False)}\n"
        status_text += f"**Torch версия:** {health_data.get('torch_version', 'unknown')}\n"
        status_text += f"**CUDA доступна:** {health_data.get('cuda_available', False)}\n"
        status_text += f"**GPU устройств:** {health_data.get('device_count', 0)}\n\n"
        
        if 'model_info' in health_data and health_data['model_info']:
            info = health_data['model_info']
            status_text += "**Информация о модели:**\n"
            status_text += f"- Класс модели: {info.get('model_class', 'unknown')}\n"
            status_text += f"- Класс токенизатора: {info.get('tokenizer_class', 'unknown')}\n"
            status_text += f"- Устройство: {info.get('device', 'unknown')}\n"
            status_text += f"- Тип данных: {info.get('dtype', 'unknown')}\n"
            status_text += f"- Имя модели: {info.get('model_name', 'unknown')}\n"
        
        if 'error' in health_data:
            status_text += f"\n⚠️ **Ошибка:** {health_data['error']}\n"
        
        return status_text
        
    except requests.exceptions.ConnectionError:
        return f"❌ Не удается подключиться к API по адресу {API_URL}\n🔧 Убедитесь, что API сервер запущен"
    except Exception as e:
        return f"❌ Ошибка получения статуса: {str(e)}"

def test_model_loading():
    """Тестирование загрузки модели через API"""
    try:
        response = requests.get(f"{API_URL}/test-model", timeout=30)
        if response.status_code == 200:
            data = response.json()
            if data.get('test_passed'):
                return "✅ Тест загрузки модели прошел успешно!"
            else:
                return f"❌ Тест провален: {data.get('error', 'Неизвестная ошибка')}"
        else:
            return f"❌ Ошибка API: {response.status_code}"
    except requests.exceptions.ConnectionError:
        return f"❌ Не удается подключиться к API по адресу {API_URL}\n🔧 Убедитесь, что API сервер запущен"
    except Exception as e:
        return f"❌ Ошибка тестирования: {str(e)}"

def generate_text(prompt, system_prompt="Вы - полезный ассистент, способный отвечать на различные вопросы."):
    """Генерация текста только через API"""
    if not prompt.strip():
        return "⚠️ Пожалуйста, введите текст для генерации"
    
    logger.info(f"📝 Генерация для: {prompt[:50]}...")
    
    try:
        logger.info("🌐 Отправка запроса к API...")
        
        response = requests.post(
            f"{API_URL}/generate", 
            json={
                "text": prompt,
                "system_prompt": system_prompt
            },
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("✅ Генерация успешна")
            return f"🤖 **Ответ модели:** {result['response']}\n\n📊 Статистика:\n- Длина входа: {result.get('input_length', 'unknown')}\n- Длина выхода: {result.get('output_length', 'unknown')}"
        else:
            error_msg = f"❌ Ошибка API: {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f"\n📋 Детали: {error_detail.get('detail', 'Нет деталей')}"
            except:
                error_msg += f"\n📋 Ответ сервера: {response.text[:200]}..."
            return error_msg
            
    except requests.exceptions.Timeout:
        return "⏰ Таймаут запроса к API (возможно, модель загружается)"
    except requests.exceptions.ConnectionError:
        return f"🔌 Ошибка подключения к API по адресу {API_URL}\n🔧 Убедитесь, что API сервер запущен"
    except Exception as e:
        logger.error(f"❌ Неожиданная ошибка API: {str(e)}")
        return f"❌ Ошибка при обращении к API: {str(e)}\n\n📋 Трейсбек: {traceback.format_exc()[:500]}..."

# Создаем интерфейс
with gr.Blocks(title="Qwen2.5-0.5B Chat UI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Qwen2.5-0.5B Chat UI")
    gr.Markdown("Чат-интерфейс для работы с моделью Qwen2.5-0.5B-Instruct")
    
    with gr.Tab("💬 Текстовый чат"):
        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    lines=3, 
                    label="Ваш вопрос", 
                    placeholder="Введите ваш вопрос здесь...",
                    value="Расскажи интересный факт о космосе"
                )
                system_prompt_text = gr.Textbox(
                    lines=2,
                    label="Системный промпт (опционально)",
                    value="Вы - полезный ассистент, способный отвечать на различные вопросы.",
                    placeholder="Настройте поведение модели..."
                )
                with gr.Row():
                    text_submit_btn = gr.Button("🚀 Отправить", variant="primary")
                    clear_btn = gr.Button("🧹 Очистить", variant="secondary")
            
            with gr.Column(scale=3):
                text_output = gr.Textbox(
                    lines=10, 
                    label="Ответ модели", 
                    interactive=False
                )
    
    with gr.Tab("🔧 Диагностика"):
        with gr.Row():
            with gr.Column():
                status_btn = gr.Button("📊 Проверить статус API", variant="primary")
                test_btn = gr.Button("🧪 Тест загрузки модели", variant="secondary")
                
            with gr.Column():
                status_output = gr.Textbox(
                    lines=15,
                    label="Статус диагностики",
                    interactive=False
                )
    
    # Информационная панель
    with gr.Accordion("ℹ️ Информация и помощь", open=False):
        gr.Markdown("""
        ### 🤖 О модели
        - **Модель**: Qwen2.5-0.5B-Instruct (494M параметров)
        - **Возможности**: Быстрая обработка текста, генерация ответов на русском и других языках
        - **Архитектура**: Transformer-based модель от Alibaba Cloud
        
        ### 🛠 Диагностические возможности
        - **Статус API**: Проверка состояния API и модели
        - **Тест модели**: Проверка загрузки модели
        - **Детальные логи**: Подробная информация об ошибках
        
        ### 🚨 Решение проблем
        - Убедитесь, что API сервер запущен на порту 8000
        - Проверьте наличие интернета для загрузки модели
        - При первом запуске загрузка модели может занять время
        
        ### 🔗 Полезные ссылки
        - [API Docs](http://localhost:8000/docs)
        - [Health Check](http://localhost:8000/health)
        - [Model Info](http://localhost:8000/model-info)
        """)
    
    # Связываем функции с интерфейсом
    text_submit_btn.click(
        fn=generate_text,
        inputs=[text_input, system_prompt_text],
        outputs=text_output
    )
    
    clear_btn.click(
        fn=lambda: ("", ""),
        inputs=[],
        outputs=[text_input, text_output]
    )
    
    status_btn.click(
        fn=get_api_status,
        inputs=[],
        outputs=status_output
    )
    
    test_btn.click(
        fn=test_model_loading,
        inputs=[],
        outputs=status_output
    )
    
    # Обработка Enter для текстового ввода
    text_input.submit(
        fn=generate_text,
        inputs=[text_input, system_prompt_text],
        outputs=text_output
    )

# Запуск с параметрами для решения проблем в Docker
if __name__ == "__main__":
    logger.info("🚀 Запуск UI...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        share=False,
        show_error=True
    )
