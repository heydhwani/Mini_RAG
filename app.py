import streamlit as st
from rag_engine import RAGEngine

st.title("🧠 RAG Document QA System")

rag = RAGEngine()

query = st.text_input("Ask a question")

if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question")

    else:

        answer, sources = rag.ask(query)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Sources")

        for s in sources:
            st.write("📄", s)