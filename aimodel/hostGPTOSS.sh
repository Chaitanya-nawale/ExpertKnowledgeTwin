#!/usr/bin/zsh
#SBATCH --account=cs-500
#SBATCH --qos=cs-500
#SBATCH --chdir /home/nawale/ExpertDigitalTwin/aimodel
#SBATCH -J hostllm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=16
#SBATCH --time=5:00:00
#SBATCH --output=logs/hostLLM%j.log

source /home/nawale/ExpertDigitalTwin/.venv/bin/activate

cd /home/nawale/ExpertDigitalTwin/aimodel

export HF_HOME=/home/nawale/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME

python3 << 'EOF'
import os
os.environ["UNSLOTH_FORCE_FA_VERSION"] = "1"
os.environ["UNSLOTH_DISABLE_WEIGHTS_INITIALIZATION"] = "1"
os.environ["TRANSFORMERS_AVOID_WEIGHT_INITIALIZATION"] = "1"

from unsloth import FastLanguageModel
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn, torch, re

MODEL_DIR = "./gpt-oss-20b-unsloth-bnb-4bit"

print("Loading model from:", MODEL_DIR)
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_DIR,
    load_in_4bit=True,
    device_map="auto"
)

# Load the model’s own chat template
with open(f"{MODEL_DIR}/chat_template.jinja", "r") as f:
    tokenizer.chat_template = f.read()

app = FastAPI()

class Query(BaseModel):
    prompt: str

def clean_output(text: str) -> str:
    # Remove multi-channel scaffolding
    text = re.sub(r'^(analysis|assistantfinal|system)[^\n]*\n?', '', text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r'<\|/?(user|assistant|system)\|>', '', text, flags=re.IGNORECASE)
    return text.strip()

@app.post("/generate")
def generate(q: Query):
    messages = [{"role": "user", "content": q.prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)

    # Slice off the prompt tokens
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    cleaned = clean_output(raw)
    return {"response": cleaned}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=16520)

EOF
