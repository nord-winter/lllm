from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def load_model():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe