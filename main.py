from fastapi import FastAPI
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI()

rag = RAGEngine()


class Query(BaseModel):
    question: str


@app.post("/ask")

def ask_question(q: Query):

    answer, sources = rag.ask(q.question)

    return {
        "question": q.question,
        "answer": answer,
        "sources": sources
    }