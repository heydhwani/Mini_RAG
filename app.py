import streamlit as st
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM



st.title("🧠 RAG Document Question Answering")

st.write("Ask questions about Artificial Intelligence, Machine Learning, and RAG.")



@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return embed_model, tokenizer, llm


embed_model, tokenizer, model_llm = load_models()



docs_path = "docs"

documents = []
doc_names = []

for file in os.listdir(docs_path):
    if file.endswith(".txt"):
        with open(os.path.join(docs_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append(text)
            doc_names.append(file)

st.write("📄 Loaded Documents:", doc_names)



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




all_chunks = []
chunk_sources = []

for i, doc in enumerate(documents):

    chunks = chunk_text(doc)

    for chunk in chunks:
        all_chunks.append(chunk)
        chunk_sources.append(i)


chunk_embeddings = embed_model.encode(all_chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")




dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(chunk_embeddings)



st.subheader("Ask a Question")

query = st.text_input("Enter your question:")


if st.button("Ask"):

    if query == "":
        st.warning("Please enter a question")

    else:

        query_embedding = embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = min(3, len(all_chunks))

        distances, indices = index.search(query_embedding, k)

        top_indices = indices[0]

        st.subheader("Retrieved Context")

        sources_used = set()

        for idx in top_indices:

            st.write(all_chunks[idx])

            source_doc = chunk_sources[idx]

            sources_used.add(doc_names[source_doc])

            st.write("Source:", doc_names[source_doc])
            st.write("---")


        # CREATE CONTEXT
        context = " ".join([all_chunks[idx] for idx in top_indices])


        prompt = f"""
You are a helpful AI assistant.

Answer the question clearly using ONLY the context.

Context:
{context}

Question:
{query}

Answer:
"""


        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


        st.subheader("Answer")
        st.write(answer)


        st.subheader("Sources")
        for s in sources_used:
            st.write("📄", s)