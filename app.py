from fastapi import FastAPI, Form
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")


@app.post("/predict")
async def predict(question: str = Form(...)):
    inputs = tokenizer(question, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=80)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return {"question": question, "answer": answer}

# Health check route
@app.get("/")
def read_root():
    return {"message": "Text-to-Text model (flan-t5-small) is running!"}
