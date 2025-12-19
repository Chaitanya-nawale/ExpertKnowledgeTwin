from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import uvicorn

app = FastAPI(title="Linq-Embed-Mistral Server")

# Load model on startup
print("Loading Linq-Embed-Mistral...")
model = SentenceTransformer("/home/nawale/models/embeddings/linq")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# Request models
class EmbeddingRequest(BaseModel):
    model: str = "Linq-Embed-Mistral"
    input: Union[str, List[str]]
    encoding_format: str = "float"

class EmbeddingResponse(BaseModel):
    object: str = "list"
    model: str
    data: list
    usage: dict

# Routes
@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def embeddings(request: EmbeddingRequest):
    try:
        texts = request.input
        
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings with GPU acceleration
        with torch.no_grad():
            embeddings = model.encode(
                texts, 
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device
            )
        
        # Format response
        data = [
            {
                "object": "embedding",
                "embedding": emb.cpu().tolist(),
                "index": i
            }
            for i, emb in enumerate(embeddings)
        ]
        
        return EmbeddingResponse(
            object="list",
            model=request.model,
            data=data,
            usage={
                "prompt_tokens": sum(len(t.split()) for t in texts),
                "total_tokens": sum(len(t.split()) for t in texts)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Linq-Embed-Mistral",
                "object": "model",
                "created": 1765299000,
                "owned_by": "Linq-AI-Research"
            }
        ]
    }

@app.get("/health")
async def health():
    return {"status": "ok", "model": "Linq-Embed-Mistral"}

@app.get("/")
async def root():
    return {
        "name": "Linq-Embed-Mistral Server",
        "version": "1.0",
        "endpoint": "/v1/embeddings"
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=16521,
        workers=1  # Single worker (GPU)
    )
