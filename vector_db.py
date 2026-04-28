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


def search_vector_db(query_embedding):
    col = client.get_collection(COLLECTION_NAME)
    out = col.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10,
        include=["documents", "distances"],
    )
    knowledge_base = out.get("documents", [[]])[0]
    distances = out.get("distances", [[]])[0]
    results = [1.0 - float(distance) for distance in distances]
    return knowledge_base, results