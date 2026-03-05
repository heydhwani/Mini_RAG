from sentence_transformers import SentenceTransformer
import numpy as np
import os
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

docs_path = "docs"
documents = []
doc_names = []

for file in os.listdir(docs_path):
    if file.endswith(".txt"):
        with open(os.path.join(docs_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

print("Documents loaded:", len(documents))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")




def chunk_text(text, chunk_size=120, overlap=20):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))

        if i + chunk_size >= len(words):
            break

    return chunks

chunk_sources = []
all_chunks = []

all_chunks = []
chunk_sources = []

for i, doc in enumerate(documents):
    chunks = chunk_text(doc)

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_sources.append(i)


print(f"\nTotal Chunks Created: {len(all_chunks)}")


chunk_embeddings = embed_model.encode(all_chunks)
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

    
    k = min(3, len(all_chunks))
    distances, indices = index.search(query_embedding, k)

    top_indices = indices[0]

    print("\nTop Relevant Chunks:")
    for idx in top_indices:
        print(f"\nChunk {idx}:")
        print(all_chunks[idx])
        source_doc = chunk_sources[idx]
        print("Source:", doc_names[source_doc])

    # Combine context
    context = " ".join([all_chunks[idx] for idx in top_indices])

    
    prompt = f"""
You are an AI tutor.

Use the context to answer the question in 3-4 clear sentences.

Context:
{context}

Question:
{query}

Answer:
"""

    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model_llm.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nGenerated Answer:\n")
    print(answer)