import PyPDF2
import os

def extract_text_from_pdfs(folder_path):
    text_data = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            pdf_file = open(path, "rb")
            reader = PyPDF2.PdfReader(pdf_file)
            for page in reader.pages:
                text_data += page.extract_text() + "\n"
    return text_data
