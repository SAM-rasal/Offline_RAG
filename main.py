import streamlit as st
from document_loader import extract_text_chunks_with_metadata
from embed_store import create_embeddings, retrieve_context_with_metadata
from rag_engine import generate_answer_with_flan_t5
import textwrap

for key in ["chunks", "metadata", "model", "index", "embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.title(" Offline RAG Document Q&A System")

if st.button("Load & Process Documents"):
    try:
        chunks, metadata = extract_text_chunks_with_metadata("data")
        st.session_state["chunks"] = chunks
        st.session_state["metadata"] = metadata
        st.success("Documents processed and chunked with metadata!")
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
            st.markdown(
                f"**File:** {meta['source_file']}  \n"
                f"**Page:** {meta['page']}  \n"
                f"**Chunk:** {meta['chunk_num']}\n"
            )
            st.text_area("Context", ctx, height=100)

