import os
import re
from pypdf import PdfReader

# --- 1. 自动定位文件夹路径 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 指向总文件夹
INPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'rawpdf')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'Law_Act', 'cleantxt')

def clean_legal_text(raw_text):
    """
    专门针对大马法律 PDF 的清洗逻辑
    """
    # 修复断句：将单换行变空格，保留双换行
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', raw_text)
    # 清除页眉页脚和页码
    text = re.sub(r'^\s*(Laws of Malaysia|ACT \d+|[0-9]+)\s*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'Page\s*\d+', '', text, flags=re.IGNORECASE)
    # 清除乱码和多余空格
    text = text.replace('\x0c', '')
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def process_all_pdfs_recursively():
    """
    深度扫描子文件夹，读取 PDF，清洗并保存
    """
    print(f"🔎 正在深度扫描目录: {INPUT_FOLDER}")
    
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 使用 os.walk 进行深度遍历
    found_any = False
    for root, dirs, files in os.walk(INPUT_FOLDER):
        for filename in files:
            if filename.lower().endswith('.pdf'):
                found_any = True
                input_path = os.path.join(root, filename)
                
                # 获取子文件夹的名字 (比如 BM version)
                subfolder_name = os.path.basename(root)
                
                # 构造输出文件名：[BM version] 原文件名.txt
                output_filename = f"[{subfolder_name}] {filename.replace('.pdf', '.txt')}"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)
                
                print(f"⏳ 发现 {subfolder_name} 中的文件: {filename}...")
                
                try:
                    reader = PdfReader(input_path)
                    full_text = ""
                    for page in reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            full_text += extracted + "\n\n"
                    
                    cleaned_text = clean_legal_text(full_text)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)
                    print(f"✅ 已洗净: {output_filename}")
                    
                except Exception as e:
                    print(f"❌ 处理失败 {filename}: {e}")

    if not found_any:
        print(f"❗ 警告：在 {INPUT_FOLDER} 及其子文件夹中没找到任何 PDF！")
        print(f"请检查路径下是否真的有文件。当前绝对路径: {os.path.abspath(INPUT_FOLDER)}")
    else:
        print(f"\n🎉 大功告成！干净的 TXT 已全部存入: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    process_all_pdfs_recursively()

