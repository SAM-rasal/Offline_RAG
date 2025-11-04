from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def create_embeddings(text_chunks):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')
    embeddings = model.encode(text_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return model, index, embeddings

def retrieve_context_with_metadata(query, model, index, chunks, metadata, top_k=3):
    q_embed = model.encode([query])
    distances, indices = index.search(q_embed, top_k)
    results = []
    for idx in indices[0]:
        results.append({
            "text": chunks[idx],
            "meta": metadata[idx]
        })
    return results
