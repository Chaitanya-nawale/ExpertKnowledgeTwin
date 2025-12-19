from sentence_transformers import SentenceTransformer

# Load the model (downloads from Hugging Face automatically)
model = SentenceTransformer("google/embeddinggemma-300m")

# Embed texts
texts = ["Sample document 1", "Sample document 2"]
embeddings = model.encode(texts, normalize_embeddings=True)

print(embeddings.shape)  # (2, 768)
