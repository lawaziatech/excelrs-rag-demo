from sentence_transformers import SentenceTransformer, util

_model = SentenceTransformer("all-MiniLM-L6-v2")

KNOWLEDGE_BASE = [
    "In a heart attack, people often report chest pain or pressure that may spread to the arm, neck, or jaw.",
    "Shortness of breath and breathlessness can occur during a heart attack but also with anxiety, asthma, or other lung problems.",
    "Migraine headaches are typically throbbing, often one-sided, and may include sensitivity to light or sound.",
    "Type 2 diabetes management often involves diet, exercise, and medications to control blood glucose.",
    "Hypertension is high blood pressure; it is often called a silent condition because it may have no obvious symptoms.",
]


def convert_to_vectors(texts: list[str]):
    """Turn a list of strings into embedding vectors (same model for chunks and query)."""
    return _model.encode(texts, convert_to_numpy=True)


def semantic_search(
    question_embedding,
    knowledge_base_embeddings,
    min_similarity: float = 0.25,
) -> list[tuple[int, float, str]]:
    """Return every chunk at or above min_similarity (no top-k: all hits that pass)."""
    results = util.cos_sim(question_embedding, knowledge_base_embeddings)[0]

    return [
        (i, float(score), KNOWLEDGE_BASE[i])
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

    print("Step 1: Convert knowledge base into vectors.")
    knowledge_base_embeddings = convert_to_vectors(KNOWLEDGE_BASE)

    print("Step 2: Convert question into vector.")
    question_embedding = convert_to_vectors([question])[0]

    print("Step 3: Semantic search to find relevant chunks.")
    chunks = semantic_search(
        question_embedding,
        knowledge_base_embeddings,
        min_similarity=0.25,
    )

    print("\nStep 4: Build context from retrieved chunks.")
    context = build_context(chunks)
    print(context)

    print("\nStep 5: Build LLM prompt from question + context.")
    prompt = make_llm_prompt(question, context)
    print(prompt)


if __name__ == "__main__":
    main()
