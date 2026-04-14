 
from utility import convert_to_vectors
from vector_db import fetch_from_vector_db
from sentence_transformers import util


def semantic_search(
    question_embedding,
    min_similarity: float = 0.25,
) -> list[tuple[int, float, str]]:
    """Return every chunk at or above min_similarity (no top-k: all hits that pass)."""

    # get knowledge base embedding from vector db
    results = search_vector_db(question_embedding)

    # results = util.cos_sim(question_embedding, knowledge_base_embeddings)[0]

    return [
        (i, float(score), knowledge_base[i])
        for i, score in enumerate(results)
        if float(score) >= min_similarity
    ]


def build_context(chunks: list[tuple[int, float, str]]) -> str:
    """Turn retrieved chunk texts into one context string for the LLM."""
    return "\n".join(f"- {text}" for _, _, text in chunks)


def make_llm_prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant. Answer using only the context below.\n"
        "If the context is empty or does not contain the answer, say: "
        "\"I do not have enough context to answer this question.\"\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def main() -> None:
    question = "Why might someone feel breathless during a possible heart attack?"

    print("Step 1: Convert question into vector.")
    question_embedding = convert_to_vectors([question])[0]  

    print("Step 3: Semantic search to find relevant chunks.")
    chunks = semantic_search(
        question_embedding,
 context from retrieved chunks.")
    context = build_context(chunks)
    print(context)

    print("\nStep 5: Build LLM prompt from question + context.")
    prompt = make_llm_prompt(question, context)
    print(prompt)


if __name__ == "__main__":
    main()
