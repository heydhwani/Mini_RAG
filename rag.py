from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Step 1: Read file
with open("data.txt", "r") as f:
    text = f.read()

chunks = text.split("\n")

chunk_embeddings = model.encode(chunks)

query = input("Ask a question: ")
