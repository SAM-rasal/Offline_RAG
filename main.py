import streamlit as st
from document_loader import extract_text_from_pdfs
from embed_store import create_embeddings, retrieve_context
from rag_engine import generate_answer_with_flan_t5
import textwrap

# --- Always initialize session state keys ---
for key in ["chunks", "model", "index", "embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("📘 Offline RAG Document Q&A System")

if st.button("Load & Process Documents"):
    try:
        text_data = extract_text_from_pdfs("data")       # make sure 'data/' exists!
        chunks = [text_data[i:i+500] for i in range(0, len(text_data), 500)]
        st.session_state["chunks"] = chunks
        st.success("Documents processed and chunked!")
    except Exception as e:
        st.error(f"Error: {e}")

if st.button("Create Embeddings"):
    if st.session_state["chunks"]:
        model, index, embeddings = create_embeddings(st.session_state["chunks"])
        st.session_state["model"] = model
        st.session_state["index"] = index
        st.session_state["embeddings"] = embeddings
        st.success("Embeddings stored in FAISS.")
    else:
        st.error("No document chunks found. Please load documents first.")

query = st.text_input("Ask your question here:")
if st.button("Generate Answer") and query:
    if not st.session_state["model"] or not st.session_state["index"] or not st.session_state["chunks"]:
        st.error("Please load documents and create embeddings first!")
    else:
        retrieved = retrieve_context(query, st.session_state["model"], st.session_state["index"], st.session_state["chunks"])
        answer = generate_answer_with_flan_t5(query, " ".join(retrieved))
        st.subheader("Generated Answer")
        st.write(textwrap.fill(answer, width=100))
        st.subheader("Retrieved Contexts")
        for ctx in retrieved:
            st.text_area("Context", ctx, height=100)
