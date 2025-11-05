import os
import streamlit as st
from document_loader import extract_text_chunks_with_metadata
from embed_store import create_embeddings, retrieve_context_with_metadata
from rag_engine import generate_answer_with_flan_t5
import textwrap

# Create a temporary folder to save uploaded PDFs
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize session state keys
for key in ["chunks", "metadata", "model", "index", "embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title("Offline RAG Document Q&A System with Upload")

# File uploader accepts multiple PDFs at once
uploaded_files = st.file_uploader(
    "Upload PDF documents", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    st.write(f"Uploading {len(uploaded_files)} files...")
    for uploaded_file in uploaded_files:
        # Save uploaded file to UPLOAD_DIR
        file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
        with open(file_path, "wb") as f_out:
            f_out.write(uploaded_file.getbuffer())
    st.success(f"Saved {len(uploaded_files)} files to {UPLOAD_DIR}!")
    
    # Process uploaded documents (extract chunks + metadata)
    chunks, metadata = extract_text_chunks_with_metadata(UPLOAD_DIR)
    st.session_state["chunks"] = chunks
    st.session_state["metadata"] = metadata
    st.success(f"Extracted {len(chunks)} chunks from uploaded documents.")

if st.button("Create Embeddings"):
    if st.session_state["chunks"]:
        model, index, embeddings = create_embeddings(st.session_state["chunks"])
        st.session_state["model"] = model
        st.session_state["index"] = index
        st.session_state["embeddings"] = embeddings
        st.success("Embeddings created and indexed.")
    else:
        st.error("No chunks found. Please upload documents first.")

query = st.text_input("Ask your question here:")

if st.button("Generate Answer") and query:
    if not st.session_state["model"] or not st.session_state["index"] or not st.session_state["chunks"]:
        st.error("Load documents and create embeddings first!")
    else:
        results = retrieve_context_with_metadata(
            query,
            st.session_state["model"], st.session_state["index"],
            st.session_state["chunks"], st.session_state["metadata"]
        )
        answer = generate_answer_with_flan_t5(query, " ".join([r["text"] for r in results]))
        st.subheader("Generated Answer")
        st.write(textwrap.fill(answer, width=100))
        st.subheader("Retrieved Contexts")
        for ctx_result in results:
            ctx = ctx_result["text"]
            meta = ctx_result["meta"]
            st.markdown(f"**File:** {meta['source_file']}  \n**Page:** {meta['page']}  \n**Chunk:** {meta['chunk_num']}")
            st.text_area("Context", ctx, height=100)
