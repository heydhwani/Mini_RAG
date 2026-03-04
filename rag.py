from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss


embed_model = SentenceTransformer('all-MiniLM-L6-v2')


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


while True:
    query = input("\nAsk a question (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting RAG system...")
        break

    # Create query embedding
    query_embedding = embed_model.encode([query])

    # Calculate similarity
    similarities = cosine_similarity(query_embedding, chunk_embeddings)

    # Get top 3 most similar chunks
    top_indices = np.argsort(similarities[0])[-3:][::-1]

    print("\nTop Relevant Chunks:")
    for idx in top_indices:
        print(f"\nScore: {similarities[0][idx]:.4f}")
        print(chunks[idx])

    
    # Apply Similarity Threshold
    
    threshold = 0.40
    filtered_indices = [
        idx for idx in top_indices
        if similarities[0][idx] > threshold
    ]

    if not filtered_indices:
        print("\nNo strong match found in context.")
        continue

    # Combine selected chunks
    context = " ".join([chunks[idx] for idx in filtered_indices])

    
    # 8️⃣ Create Prompt
    
    prompt = f"""
You are a helpful AI assistant.

Answer the question in a complete sentence.
Use ONLY the information from the context.
Do not give short or one-word answers.

Context:
{context}

Question: {query}

Answer:
"""

    
    # Generate Answer
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model_llm.generate(
        **inputs,
        max_new_tokens=80,
        min_new_tokens=10,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:\n")
    print(answer)

    
    # Show Source Chunks
    
    print("\nSource Chunks Used:")
    for idx in filtered_indices:
        print(f"- Chunk {idx}")