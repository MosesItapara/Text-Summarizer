from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
summarizer = pipeline("summarization", model="artifacts/model_trainer")

class TextInput(BaseModel):
    text: str

@app.post("/summarize")
def summarize(payload: TextInput):
    result = summarizer(payload.text, max_length=128, min_length=30)
    return {"summary": result[0]["summary_text"]}