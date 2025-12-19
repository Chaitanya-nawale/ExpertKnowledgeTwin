from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import uvicorn

app = FastAPI(title="Ministral-3 14B 8-bit API")

MODEL_NAME = "mistralai/Ministral-3-14B-Instruct-2512"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Ministral-3-14B-Instruct-2512",
    trust_remote_code=True,
    use_fast=False,  # IMPORTANT: Tekken is a Python tokenizer
)

print("Loading model in 8-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True,
)

model.eval()
print("Model loaded successfully!")

class RequestBody(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.post("/generate")
async def generate(req: RequestBody):

    inputs = tokenizer(
        req.prompt,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=req.max_tokens,
            do_sample=True,
            temperature=0.7,
        )

    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return {"output": text}

@app.get("/")
def root():
    return {"status": "running"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=16520)
