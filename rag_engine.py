from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np
import os
import faiss


class RAGEngine:

    def __init__(self, docs_path="docs"):

        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
        self.model_llm = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base").to("cpu")

        documents = []
        doc_names = []

        for file in os.listdir(docs_path):
            if file.endswith(".txt"):
                with open(os.path.join(docs_path, file), "r", encoding="utf-8") as f:
                    documents.append(f.read())
                    doc_names.append(file)

        self.doc_names = doc_names

        self.all_chunks = []
        self.chunk_sources = []

        for i, doc in enumerate(documents):
            chunks = self.chunk_text(doc)

            for chunk in chunks:
                self.all_chunks.append(chunk)
                self.chunk_sources.append(i)

        embeddings = self.embed_model.encode(self.all_chunks, normalize_embeddings=True)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)


    def chunk_text(self, text, chunk_size=120, overlap=20):

        words = text.split()
        chunks = []

        step = chunk_size - overlap

        for i in range(0, len(words), step):

            chunk = words[i:i + chunk_size]
            chunks.append(" ".join(chunk))

            if i + chunk_size >= len(words):
                break

        return chunks


    def ask(self, question):

        query_embedding = self.embed_model.encode([question], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")

        k = min(3, len(self.all_chunks))

        distances, indices = self.index.search(query_embedding, k)

        top_indices = indices[0]

        context = " ".join([self.all_chunks[idx] for idx in top_indices])

        sources = list(set([self.doc_names[self.chunk_sources[idx]] for idx in top_indices]))

        prompt = f"""
Answer the question using the given context.

Context:
{context}

Question:
{question}

Write a clear explanation in 3 sentences.
"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to("cpu")

        outputs = self.model_llm.generate(
            **inputs,
            max_new_tokens=150,
            num_beams=4,
            early_stopping=True
        )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return answer, sources