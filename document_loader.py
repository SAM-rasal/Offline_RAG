import os
import PyPDF2

def extract_text_chunks_with_metadata(folder_path, chunk_size=500):
    chunks = []
    metadata = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            with open(path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page_no, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    for i in range(0, len(page_text), chunk_size):
                        chunk = page_text[i:i+chunk_size]
                        chunks.append(chunk)
                        metadata.append({
                            "source_file": file,
                            "page": page_no,
                            "chunk_num": i//chunk_size + 1
                        })
    return chunks, metadata
