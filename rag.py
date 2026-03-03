from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import pipeline

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Read file
with open("data.txt", "r") as f:
    text = f.read()

def chunk_text(text, chunk_size=10, overlap=2):
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

print("\nGenerated Chunks:")
for idx, chunk in enumerate(chunks):
    print(f"\nChunk {idx}:\n{chunk}")


chunks = chunk_text(text)

chunk_embeddings = model.encode(chunks)

query = input("Ask a question: ")

query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, chunk_embeddings)

top_indices = np.argsort(similarities[0])[-3:][::-1]

print("\nTop 3 Relevant Chunks:")
for idx in top_indices:
    print(f"\nScore: {similarities[0][idx]:.4f}")
    print(chunks[idx])

generator = pipeline("text2text-generation", model="google/flan-t5-small")

context = " ".join([chunks[idx] for idx in top_indices])

prompt = f"""
Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer:
"""

response = generator(prompt, max_length=150)

print(response[0]["generated_text"])