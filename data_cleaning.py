import os
import re
import docx
from pypdf import PdfReader

# --- 1. Automatically locate the current script's directory ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 Path configuration: Corrected to point to the subdirectories under the Law_Act folder
INPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'rawpdf')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'cleantxt')

def clean_legal_text(raw_text):
    """
    Cleaning logic specifically tailored for Malaysian legal texts
    """
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text)
    text = re.sub(r'^\s*(Laws of Malaysia|ACT \d+|[0-9]+)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
    text = text.replace('\x0c', '') 
    text = text.replace('•', '-')   
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_all_files():
    """
    Scan the Law_Act/rawpdf folder, process both PDF and Word files, and save to Law_Act/cleantxt
    """
    print(f"🚀 Precisely scanning: {INPUT_FOLDER}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    found_any = False
    
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            full_text = ""
            output_filename = ""
            input_path = os.path.join(root, filename)
            
            if filename.lower().endswith('.pdf'):
                found_any = True
                print(f"⏳ Extracting PDF: {filename}...")
                try:
                    reader = PdfReader(input_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n\n"
                    output_filename = re.sub(r'\.pdf$', '.txt', filename, flags=re.IGNORECASE)
                except Exception as e:
                    print(f"❌ Failed to read PDF {filename}: {e}")
                    continue

            elif filename.lower().endswith('.docx'):
                found_any = True
                print(f"⏳ Extracting Word: {filename}...")
                try:
                    doc = docx.Document(input_path)
                    for para in doc.paragraphs:
                        full_text += para.text + "\n"
                    output_filename = re.sub(r'\.docx$', '.txt', filename, flags=re.IGNORECASE)
                except Exception as e:
                    print(f"❌ Failed to read Word {filename}: {e}")
                    continue

            if output_filename:
                cleaned_data = clean_legal_text(full_text)
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_data)
                print(f"✅ Cleaned: {output_filename}")

    if not found_any:
        print(f"❗ Error: No files found in {INPUT_FOLDER}!")
        print(f"Please check if there are actually .pdf or .docx files in this directory.")
    else:
        print(f"\n🎉 Cleaning complete! Please check: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_all_files()
