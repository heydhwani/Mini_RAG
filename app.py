import streamlit as st
import os
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="AI Knowledge Assistant", page_icon="🧠")


st.title("🧠 AI Knowledge Assistant")
st.write("Ask questions about Artificial Intelligence, Machine Learning, and RAG.")


# Sidebar
st.sidebar.header("ℹ️ Instructions")
st.sidebar.write(
"""
1️⃣ Ask a question related to AI or ML  
2️⃣ The system retrieves relevant knowledge  
3️⃣ AI generates a clear answer
"""
)

st.sidebar.header("💡 Example Questions")
example_questions = [
"What is Machine Learning?",
"Explain Artificial Intelligence",
"What is Retrieval Augmented Generation?",
"Difference between AI and ML?"
]

for q in example_questions:
    st.sidebar.write("•", q)


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


st.subheader("❓ Ask a Question")

query = st.text_input("Type your question here")


if st.button("Get Answer"):

    if query == "":
        st.warning("Please enter a question")

    else:

        query_embedding = embed_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        k = min(3, len(all_chunks))

        distances, indices = index.search(query_embedding, k)

        top_indices = indices[0]


        context = " ".join([all_chunks[idx] for idx in top_indices])


        prompt = f"""
You are a helpful AI assistant.

Answer clearly using the provided context.

Context:
{context}

Question:
{query}

Answer:
"""


        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

        outputs = model_llm.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)


        st.subheader("💡 Answer")
        st.success(answer)