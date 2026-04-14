from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def convert_to_vectors(texts: list[str]):
    """Turn a list of strings into embedding vectors (same model for chunks and query)."""
    return _model.encode(texts, convert_to_numpy=True)