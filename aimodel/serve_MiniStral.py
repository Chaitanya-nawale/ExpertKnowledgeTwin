from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import uvicorn

app = FastAPI(title="8-bit 8B Model API")

# Model config
MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"

print("Loading 8-bit model...")

# Use BitsAndBytesConfig for 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)
model.eval()

print("Model loaded successfully!")

# Request body
class RequestBody(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(request: RequestBody):
    inputs = tokenizer(request.prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            temperature=0.7
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"output": text}

@app.get("/")
async def root():
    return {"status": "running"}

# Run server on 0.0.0.0:16520
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=16520, log_level="info")
