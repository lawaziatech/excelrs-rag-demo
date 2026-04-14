from utility import convert_to_vectors
from vector_db import save_to_vector_db

def load_knowledge_base():
    with open("knowledge_base.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file]



def main():
    print("Step 1: load knowledge base") 
    knowledge_base = load_knowledge_base()

    print("Step 2: convert to vectors")
    knowledge_base_embeddings = convert_to_vectors(knowledge_base)

    print("Step 3: save to vector db")
    save_to_vector_db(knowledge_base_embeddings)

if __name__ == "__main__":
    main()
