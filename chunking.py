import os
# 【改造一】：我们不再用 unstructured 读 PDF，而是用 LangChain 的文本切块器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. 定义干净文本的来源文件夹
CLEAN_TXT_FOLDER = './Law_Act/cleantxt'

# 2. 初始化一个强大的文本切块器
# 这个切块器会自动按段落、句子、单词来切，非常智能
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,    # 【可调参数】每个 chunk 的最大字数，建议 1500-2500
    chunk_overlap=200,  # 【可调参数】每个 chunk 之间重叠的字数，防止把重要句子从中间切断
    length_function=len,
    is_separator_regex=False,
)

# 3. 创建一个空列表，用来存放所有切好的 chunks
all_chunks = []

def create_chunks_from_folder():
    """
    遍历文件夹，读取干净的 TXT，切块并注入元数据。
    """
    print(f"🚀 开始从 {CLEAN_TXT_FOLDER} 文件夹读取干净文本并进行切块...")
    
    txt_files = [f for f in os.listdir(CLEAN_TXT_FOLDER) if f.endswith('.txt')]
    
    if not txt_files:
        print("❌ 没有找到干净的 TXT 文件！请先运行 data_cleaning.py。")
        return []

    for filename in txt_files:
        file_path = os.path.join(CLEAN_TXT_FOLDER, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                full_text = f.read()
                
            # 【改造二】：使用 text_splitter 对每个文件进行切块
            # 它会返回一个 Document 对象的列表
            file_chunks = text_splitter.create_documents([full_text])
            
            # 【改造三】：为每个 chunk 注入“来源”元数据！(极其重要)
            for chunk in file_chunks:
                # 这一步告诉 AI，这段文字来自哪本书
                chunk.metadata["source_document"] = filename 
            
            all_chunks.extend(file_chunks) # 把当前文件的所有 chunks 加入总列表
            
            print(f"✅ 文件 {filename} 已成功切成 {len(file_chunks)} 块。")
            
        except Exception as e:
            print(f"❌ 读取或切块 {filename} 失败，错误信息: {e}")
            
    print(f"🎉 所有 TXT 文件处理完毕！总共生成了 {len(all_chunks)} 个知识块。")
    return all_chunks

# 主程序：调用函数并把结果存入一个变量，供其他文件 import
chunks = create_chunks_from_folder()

# (可选) 测试一下，看看第一个 chunk 长什么样
if __name__ == "__main__":
    if chunks:
        print("\n--- 第一个 Chunk 预览 ---")
        print(f"内容: {chunks[0].page_content[:300]}...") # 打印前300个字
        print(f"元数据: {chunks[0].metadata}")