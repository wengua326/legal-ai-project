import os
import re
import docx
from pypdf import PdfReader

# --- 1. 自动定位当前脚本所在的目录 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 💡 路径配置：修正为指向 Law_Act 文件夹下的子目录
INPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'rawpdf')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'cleantxt')

def clean_legal_text(raw_text):
    """
    专门针对大马法律文本的清洗逻辑
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
    扫描 Law_Act/rawpdf 文件夹，同时处理 PDF 和 Word，保存到 Law_Act/cleantxt
    """
    print(f"🚀 正在精准扫描: {INPUT_FOLDER}")
    
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
                print(f"⏳ 正在榨取 PDF: {filename}...")
                try:
                    reader = PdfReader(input_path)
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n\n"
                    output_filename = re.sub(r'\.pdf$', '.txt', filename, flags=re.IGNORECASE)
                except Exception as e:
                    print(f"❌ PDF 读取失败 {filename}: {e}")
                    continue

            elif filename.lower().endswith('.docx'):
                found_any = True
                print(f"⏳ 正在榨取 Word: {filename}...")
                try:
                    doc = docx.Document(input_path)
                    for para in doc.paragraphs:
                        full_text += para.text + "\n"
                    output_filename = re.sub(r'\.docx$', '.txt', filename, flags=re.IGNORECASE)
                except Exception as e:
                    print(f"❌ Word 读取失败 {filename}: {e}")
                    continue

            if output_filename:
                cleaned_data = clean_legal_text(full_text)
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_data)
                print(f"✅ 已洗净: {output_filename}")

    if not found_any:
        print(f"❗ 错误：在 {INPUT_FOLDER} 没找到文件！")
        print(f"请检查该路径下是否真的有 .pdf 或 .docx 文件。")
    else:
        print(f"\n🎉 清洗完成！请查看: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_all_files()