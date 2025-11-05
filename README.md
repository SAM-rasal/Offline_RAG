Offline Retrieval-Augmented Generation (RAG) System
Overview
This project implements an end-to-end offline RAG pipeline that enables question answering over PDF documents without internet access. 
It extracts and chunks PDF text with metadata, creates semantic embeddings, and uses a local language model for grounded answer generation.


Features
Upload your own PDF documents via a Streamlit interface.
Text extraction with per-chunk metadata including source file and page number.
Embedding generation using pre-trained MiniLM SentenceTransformers.
Fast similarity search with FAISS vector store.
Answer generation using local Flan-T5 model.
Evaluation on custom datasets with BLEU, ROUGE-L, BERTScore, and Cosine similarity.
Displays retrieved context along with PDF source and page/chunk info.


Technologies
Python
PyPDF2 (PDF text extraction)
SentenceTransformers (MiniLM embeddings)
FAISS (fast approximate nearest neighbor search)
HuggingFace Transformers (Flan-T5 LLM)
Streamlit (interactive web app)
NLTK, scikit-learn, rouge-score, bert-score (evaluation metrics)
