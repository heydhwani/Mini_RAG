import streamlit as st
from rag_engine import RAGEngine

st.set_page_config(page_title="RAG Document QA System", page_icon="🧠")

st.title("🧠 RAG Document QA System")
st.write("Ask questions about Artificial Intelligence, Machine Learning, and RAG.")


# ---------- SIDEBAR ----------

st.sidebar.header("ℹ️ Instructions")

st.sidebar.write("""
1️⃣ Enter a question about AI or Machine Learning  
2️⃣ Click **Ask**  
3️⃣ The system retrieves relevant document chunks  
4️⃣ AI generates an answer from the documents
""")

st.sidebar.header("💡 Example Questions")

st.sidebar.write("• What is Artificial Intelligence?")
st.sidebar.write("• What is Machine Learning?")
st.sidebar.write("• What is Retrieval Augmented Generation?")
st.sidebar.write("• Difference between AI and ML?")


# ---------- LOAD RAG ENGINE ----------

@st.cache_resource
def load_rag():
    return RAGEngine()

rag = load_rag()


# ---------- MAIN INPUT ----------

st.subheader("❓ Ask a Question")

query = st.text_input("Type your question here")


if st.button("Ask"):

    if query.strip() == "":
        st.warning("Please enter a question")

    else:

        with st.spinner("Generating answer..."):

            answer, sources = rag.ask(query)

        st.subheader("💡 Answer")
        st.success(answer)


        st.subheader("📄 Sources")

        for s in sources:
            st.write("•", s)