from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
import uvicorn
import httpx
import json

app = FastAPI(title="Expert Digital Twin")

OPENAI_BACKEND_URL = "http://i28:16520/v1/chat/completions"

print("Loading Linq-Embed-Mistral...")
embedding_model = SentenceTransformer("/home/nawale/models/embeddings/linq")

class EmbeddingRequest(BaseModel):
    model: str = "Linq-Embed-Mistral"
    input: Union[str, List[str]]

@app.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    try:
        texts = request.input if isinstance(request.input, list) else [request.input]
        with torch.no_grad():
            embeddings = embedding_model.encode(
                texts, 
                normalize_embeddings=True,
                convert_to_tensor=True
            )
        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "embedding": emb.cpu().tolist(),
                    "index": i
                }
                for i, emb in enumerate(embeddings)
            ],
            "model": "Linq-Embed-Mistral",
            "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "Ministral-3-14B-Instruct-2512",
                "object": "model",
                "created": 1765299000,
                "owned_by": "mistralai",
                "permission": [],
                "root": "Ministral-3-14B-Instruct-2512",
                "parent": None
            },
            {
                "id": "Linq-Embed-Mistral",
                "object": "model",
                "created": 1765299000,
                "owned_by": "linq-ai-research",
                "permission": [],
                "root": "Linq-Embed-Mistral",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def chat(request: Request):
    try:
        payload = await request.json()
    except Exception as e:
        return JSONResponse(content={"error": f"Invalid JSON: {str(e)}"}, status_code=400)

    stream_mode = payload.get("stream", False)

    if stream_mode:
        async def event_generator():
            client = httpx.AsyncClient(timeout=None)
            try:
                async with client.stream("POST", OPENAI_BACKEND_URL, json=payload) as r:
                    if r.status_code != 200:
                        error_text = await r.aread()
                        yield f"data: {json.dumps({'error': error_text.decode()})}\n\n"
                        return
                    
                    async for line in r.aiter_lines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("data: "):
                            yield f"{line}\n\n"
                        else:
                            yield f"data: {line}\n\n"
                    yield "data: [DONE]\n\n"
                    
            except httpx.StreamClosed:
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            finally:
                await client.aclose()

        return StreamingResponse(
            event_generator(), 
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        async with httpx.AsyncClient(timeout=None) as client:
            resp = await client.post(OPENAI_BACKEND_URL, json=payload)
            try:
                return JSONResponse(content=resp.json())
            except Exception:
                return JSONResponse(content={"error": "Invalid response from backend"}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=16522, workers=1)
