import streamlit as st
from document_loader import extract_text_from_pdfs
from embed_store import create_embeddings, retrieve_context
from rag_engine import generate_answer_with_flan_t5
import textwrap

st.title("📘 Offline RAG Document Q&A System")

if st.button("Load & Process Documents"):
    text_data = extract_text_from_pdfs("data")       # make sure 'data/' exists!
    chunks = [text_data[i:i+500] for i in range(0, len(text_data), 500)]
    st.session_state["chunks"] = chunks
    st.success("Documents processed and chunked!")

if st.button("Create Embeddings"):
    model, index, embeddings = create_embeddings(st.session_state["chunks"])
    st.session_state["model"] = model
    st.session_state["index"] = index
    st.success("Embeddings stored in FAISS.")

query = st.text_input("Ask your question here:")
if st.button("Generate Answer") and query:
    retrieved = retrieve_context(query, st.session_state["model"], st.session_state["index"], st.session_state["chunks"])
    answer = generate_answer_with_flan_t5(query, " ".join(retrieved))
    st.subheader("Generated Answer")
    st.write(textwrap.fill(answer, width=100))
    st.subheader("Retrieved Contexts")
    for ctx in retrieved:
        st.text_area("Context", ctx, height=100)
