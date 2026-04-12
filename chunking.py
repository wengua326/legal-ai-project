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
    
    print(f"🚀 开始从 {CLEAN_TXT_FOLDER} 文件夹读取干净文本并进行切块...")
    
    txt_files = [f for f in os.listdir(CLEAN_TXT_FOLDER) if f.endswith('.txt')]
    
    if not txt_files:
        print("❌ 没有找到干净的 TXT 文件！")
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
            print(f"❌ 读取或切块 {filename} 失败，错误信息: {e}")
            
    print(f"🎉 所有 TXT 文件处理完毕！总共生成了 {len(all_chunks)} 个知识块。")
    return all_chunks

# ✂️ 【修改点】：删除了底部的 chunks = create_chunks_from_folder()，防止 import 时自动运行