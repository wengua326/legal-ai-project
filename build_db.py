import os
import uuid
import pickle
from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv()

# 导入你的核心模块
import chunking as ck
import summary as sm
import embedding as emb

def build_database():
    print("=========================================")
    print("🚀 MyLegal 数据库构建程序 [终极健壮版]")
    print("=========================================\n")

    backup_file = "mylegal_backup_update_v2.pkl"
    my_chunks = []
    my_summaries = []

    # --- Step 1: 加载数据 (优先从备份加载) ---
    if os.path.exists(backup_file):
        print(f"📂 发现本地备份文件 '{backup_file}'，正在加载...")
        try:
            with open(backup_file, "rb") as f:
                data = pickle.load(f)
                my_chunks = data['chunks']
                my_summaries = data['summaries']
            print(f"✅ 加载成功！原始条数: {len(my_chunks)}")
        except Exception as e:
            print(f"❌ 备份文件读取失败: {e}")
            return
    else:
        print("⚠️ 未检测到备份，开始全新的切块与 API 总结逻辑...")
        my_chunks = ck.create_chunks_from_folder()
        if not my_chunks:
            print("❌ 未能成功读取 TXT 文件。")
            return
        my_summaries = sm.generate_summaries(my_chunks)
        
        # 实时保存备份
        print("\n💾 正在保存备份文件以防万一...")
        with open(backup_file, "wb") as f:
            pickle.dump({'chunks': my_chunks, 'summaries': my_summaries}, f)

    # --- Step 2: 数据清洗与体检 (防止 400 错误) ---
    print(f"\n🩺 正在对数据进行入库前体检，剔除无效内容...")
    
    summary_docs = []
    final_ids = []
    final_chunks = []
    
    for i, summary_text in enumerate(my_summaries):
        # 🌟 核心过滤：剔除 None, 空字符串, 全空格, 或 [Error Summary]
        if summary_text and str(summary_text).strip() and summary_text != "[Error Summary]":
            source_file = my_chunks[i].metadata.get("source_document", "Unknown_Source")
            
            # 生成这一对数据的唯一钥匙
            u_id = str(uuid.uuid4())
            
            # 包装成 ChromaDB 认识的格式
            doc = Document(
                page_content=str(summary_text).strip(),
                metadata={
                    'doc_id': u_id,       
                    'source': source_file       
                }
            )
            summary_docs.append(doc)
            final_ids.append(u_id)
            final_chunks.append(my_chunks[i])
            
    skipped_count = len(my_chunks) - len(summary_docs)
    print(f"✅ 体检完毕！")
    if skipped_count > 0:
        print(f"⚠️ 自动过滤了 {skipped_count} 条会导致报错的空摘要。")
    print(f"📦 最终有效入库条数: {len(summary_docs)}")

    # --- Step 3: 分批写入数据库 ---
    if not summary_docs:
        print("❌ 没有任何有效数据可以存入数据库。")
        return

    print("\n🚧 正在分批写入数据库 (ChromaDB & Docstore)...")
    batch_size = 500  
    total_len = len(summary_docs)
    
    

    for i in tqdm(range(0, total_len, batch_size), desc="入库进度"):
        end_idx = min(i + batch_size, total_len)
        
        curr_batch_docs = summary_docs[i:end_idx]
        curr_batch_ids = final_ids[i:end_idx]
        curr_batch_chunks = final_chunks[i:end_idx]
        
        try:
            # 1. 存入向量库 (ChromaDB 会自动调用 Embedding API)
            emb.vectorstore.add_documents(curr_batch_docs)
            
            # 2. 存入原始文档库 (Docstore)
            batch_bytes = [pickle.dumps(c) for c in curr_batch_chunks]
            emb.store.mset(list(zip(curr_batch_ids, batch_bytes)))
            
        except Exception as e:
            print(f"\n❌ 在批次 {i} 到 {end_idx} 处写入失败: {e}")
            print("💡 建议：如果持续报错，请尝试将 batch_size 调小到 100。")
            return

    print("\n" + "="*40)
    print("🎉 任务圆满完成！大马法律大脑构建成功！")
    print(f"📁 向量库: ./chroma_db")
    print(f"📁 原始库: ./docstore")
    print("="*40)

if __name__ == "__main__":
    build_database()