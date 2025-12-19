from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from typing import List
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="EmbeddingGemma Server", version="1.0.0")

# Initialize model globally (loaded once at startup)
model = None

class EmbedRequest(BaseModel):
    texts: List[str]
    embedding_dim: int = 768  # Optional: truncate to smaller dimensions

class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int

@app.on_event("startup")
async def load_model():
    """Load the model at startup"""
    global model
    try:
        logger.info("Loading EmbeddingGemma-300M model...")
        hf_token = os.getenv("HF_TOKEN")
        model = SentenceTransformer(
            "google/embeddinggemma-300m",
            token=hf_token,
            device="cuda"  # Use GPU if available, else CPU
        )
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model
    if model:
        del model
        logger.info("Model unloaded")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "google/embeddinggemma-300m"
    }

@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """
    Generate embeddings for a list of texts
    
    Args:
        request: EmbedRequest with texts and optional embedding_dim
    
    Returns:
        EmbedResponse with embeddings and dimension
    """
    global model
    
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Check server logs."
        )
    
    if not request.texts:
        raise HTTPException(
            status_code=400,
            detail="texts list cannot be empty"
        )
    
    if len(request.texts) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 texts per request"
        )
    
    try:
        # Generate embeddings
        embeddings = model.encode(
            request.texts,
            normalize_embeddings=True,
            convert_to_numpy=False  # Keep as tensor for better performance
        )
        
        # Truncate dimensions if requested (Matryoshka representation learning)
        if request.embedding_dim < 768:
            embeddings = embeddings[:, :request.embedding_dim]
        
        # Convert to list of lists
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
        
        return EmbedResponse(
            embeddings=embeddings_list,
            dimension=request.embedding_dim
        )
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )

@app.post("/embed-batch")
async def embed_batch(request: EmbedRequest):
    """
    Generate embeddings with batch processing for large datasets
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(request.texts), batch_size):
            batch = request.texts[i:i + batch_size]
            embeddings = model.encode(
                batch,
                normalize_embeddings=True,
                convert_to_numpy=False
            )
            
            if request.embedding_dim < 768:
                embeddings = embeddings[:, :request.embedding_dim]
            
            all_embeddings.extend(embeddings.tolist())
        
        return EmbedResponse(
            embeddings=all_embeddings,
            dimension=request.embedding_dim
        )
    
    except Exception as e:
        logger.error(f"Error in batch processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "EmbeddingGemma-300M Server",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /embed": "Generate embeddings for texts",
            "POST /embed-batch": "Batch generate embeddings"
        }
    }
