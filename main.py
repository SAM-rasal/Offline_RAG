import os
import PyPDF2
from PIL import Image
import pytesseract

def extract_text_chunks_with_metadata(folder_path, chunk_size=500, tesseract_cmd=None):
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    chunks = []
    metadata = []
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.lower().endswith(".pdf"):
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
                            "chunk_num": i//chunk_size + 1,
                            "source_type": "pdf"
                        })
        elif file.lower().endswith((".png", ".jpg", ".jpeg")):
            try:
                image = Image.open(path)
                text = pytesseract.image_to_string(image)
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size]
                    chunks.append(chunk)
                    metadata.append({
                        "source_file": file,
                        "page": 1,
                        "chunk_num": i//chunk_size + 1,
                        "source_type": "image"
                    })
            except Exception as e:
                continue
    return chunks, metadata
