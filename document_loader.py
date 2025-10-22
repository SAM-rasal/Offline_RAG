import os
import PyPDF2

def extract_text_from_pdfs(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder '{folder_path}' does not exist. Please create it and add your PDFs.")
    text_data = ""
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            with open(path, "rb") as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                for page in reader.pages:
                    text_data += page.extract_text() + "\n"
    return text_data
