from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


embed_model = SentenceTransformer("all-MiniLM-L6-v2")


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()


def chunk_text(text, chunk_size=40, overlap=5):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

        if i + chunk_size >= len(words):
            break

    return chunks


chunks = chunk_text(text)

print(f"\nTotal Chunks Created: {len(chunks)}")


chunk_embeddings = embed_model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")


dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)

# Add embeddings to index
index.add(chunk_embeddings)

print("FAISS index created and embeddings stored.")


while True:

    query = input("\nAsk a question (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting RAG system...")
        break

    
    query_embedding = embed_model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    
    k = min(3, len(chunks))
    distances, indices = index.search(query_embedding, k)

    top_indices = indices[0]

    print("\nTop Relevant Chunks:")
    for idx in top_indices:
        print(f"\nChunk {idx}:")
        print(chunks[idx])

    # Combine context
    context = " ".join([chunks[idx] for idx in top_indices])

    
    prompt = f"""
You are a helpful AI assistant.

Answer the question clearly using ONLY the context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {query}

Answer:
"""

    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model_llm.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:\n")
    print(answer)