--- START OF FILE test132.py ---

import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

CLEAN_TXT_FOLDER = './Law_Act/cleantxt'

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    
    chunk_overlap=200,  
    length_function=len,
    is_separator_regex=False,
)

def create_chunks_from_folder():
    all_chunks =[] 
    
    print(f"🚀 Starting to read clean text from the {CLEAN_TXT_FOLDER} folder and perform chunking...")
    
    txt_files = [f for f in os.listdir(CLEAN_TXT_FOLDER) if f.endswith('.txt')]
    
    if not txt_files:
        print("❌ No clean TXT files found!")
        return[]

    for filename in txt_files:
        file_path = os.path.join(CLEAN_TXT_FOLDER, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
                
            file_chunks = text_splitter.create_documents([full_text])
            
            for chunk in file_chunks:
                chunk.metadata["source_document"] = filename 
            
            all_chunks.extend(file_chunks) 
            
        except Exception as e:
            print(f"❌ Failed to read or chunk {filename}. Error: {e}")
            
    print(f"🎉 All TXT files processed successfully! Generated a total of {len(all_chunks)} knowledge chunks.")
    return all_chunks
