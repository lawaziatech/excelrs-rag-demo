import chromadb
import numpy as np

COLLECTION_NAME = "knowledge_base"
CHROMA_FOLDER = "."
client = chromadb.PersistentClient(path=CHROMA_FOLDER)

def fetch_from_vector_db():
    col = client.get_collection(COLLECTION_NAME)
    out = col.get(include=["documents", "embeddings"])
    return list(out["documents"]), np.array(out["embeddings"], dtype=np.float32)
    #implement fetch from vector db

def save_to_vector_db(knowledge_base, knowledge_base_embeddings):
    col = client.get_or_create_collection(COLLECTION_NAME)
    col.add(
        ids=[str(i) for i in range(len(knowledge_base_embeddings))],
        documents=knowledge_base,
        embeddings=knowledge_base_embeddings,
    )

def search_vector_db(embeddings):
    col = client.get_collection(COLLECTION_NAME)
    results = col.query(
        query_embeddings=[embeddings],
        n_results=10,
    )
    return results