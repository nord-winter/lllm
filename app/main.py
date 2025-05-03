from fastapi import FastAPI
from pydantic import BaseModel
from app.model import load_model

app = FastAPI()
generator = load_model()

class Prompt(BaseModel):
    text: str

@app.post("/generate")
def generate(prompt: Prompt):
    output = generator(prompt.text, max_new_tokens=100)[0]["generated_text"]
    return {"response": output}