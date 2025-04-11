from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = FastAPI()

# Load the FLAN-T5 model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

@app.post("/predict")
async def predict(question: str = Form(...)):
    try:
        # Tokenize and generate the response
        inputs = tokenizer(question, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=80)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"question": question, "answer": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# Health check for GET requests
@app.get("/")
def read_root():
    return {"message": "Text-to-Text model is running!"}

# Health check for HEAD requests (Render specific)
@app.head("/")
def read_root_head():
    return {"message": "OK"}
