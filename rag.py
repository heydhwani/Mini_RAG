from rag_engine import RAGEngine

rag = RAGEngine()

while True:

    query = input("\nAsk a question (type 'exit' to quit): ").strip()

    if query.lower() == "exit":
        print("Exiting RAG system...")
        break

    answer, sources = rag.ask(query)

    print("\nGenerated Answer:\n")
    print(answer)

    print("\nSources:")
    for s in sources:
        print(s)