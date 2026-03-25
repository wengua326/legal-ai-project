import os
import uuid
import pickle
from dotenv import load_dotenv
from langchain_core.documents import Document

# 导入我们之前写好的模块
import chunking as ck
import summary as sm
import embedding as emb

# 加载环境变量 (比如你的 API Key)
load_dotenv()

def build_database():
    """
    将生成的 Summary (存入 ChromaDB) 和原始 Chunk (存入 LocalFileStore) 绑定并入库。
    """
    print("🚀 开始构建/更新法律知识库...")

    # 1. 检查是否有数据需要入库
    # ck.chunks 是原始的文本块，sm.text_summaries 是对应的摘要
    if not ck.chunks or not sm.text_summaries:
        print("❌ 没有检测到需要入库的数据。请检查前置步骤！")
        return

    # 确保原始块和摘要的数量是一一对应的
    if len(ck.chunks) != len(sm.text_summaries):
        print("❌ 严重错误：Chunk 的数量与 Summary 的数量不匹配！")
        return

    print(f"📦 准备将 {len(ck.chunks)} 对[摘要-原文]存入数据库...")

    # 2. 为每一对数据生成一个全球唯一的 ID (UUID)
    # 这就像给每一本书发一个身份证号，ChromaDB 和 LocalFileStore 就靠这个号码相认
    doc_ids = [str(uuid.uuid4()) for _ in ck.chunks]

    # 3. 准备存入 ChromaDB 的数据 (摘要)
    # 我们把 Summary 包装成 Document 对象，把刚才生成的 ID 和【来源文件名】塞进 metadata 里
    summary_docs =[]
    for i, summary_text in enumerate(sm.text_summaries):
        # 从原始 chunk 中提取来源文件名 (我们在 chunking.py 里加的)
        source_file = ck.chunks[i].metadata.get("source_document", "Unknown_Source")
        
        doc = Document(
            page_content=summary_text,
            metadata={
                'doc_id': doc_ids[i],       # 绑定 ID
                'source': source_file       # 绑定来源，告诉 AI 这是哪本法律！
            }
        )
        summary_docs.append(doc)

    # 4. 执行入库：存入 ChromaDB (向量库)
    print("⏳ 正在将摘要向量化并存入 ChromaDB...")
    try:
        emb.retriever.vectorstore.add_documents(summary_docs)
    except Exception as e:
        print(f"❌ ChromaDB 入库失败: {e}")
        return

    # 5. 准备存入 LocalFileStore 的数据 (原始 Chunk)
    # 因为原始 Chunk 可能很大，包含各种信息，LocalFileStore 要求存 Byte 格式，所以用 pickle 冻结它
    print("⏳ 正在将原始法条存入本地文档库 (Docstore)...")
    try:
        # 将原始 Chunk 对象转换为 Byte
        chunk_bytes = [pickle.dumps(chunk) for chunk in ck.chunks]
        
        # 将 ID 和 Byte 数据打包成键值对 (zip)，并存入 Docstore (mset)
        emb.retriever.docstore.mset(list(zip(doc_ids, chunk_bytes)))
    except Exception as e:
        print(f"❌ Docstore 入库失败: {e}")
        return

    print("🎉 数据库构建/更新成功！")

# 运行主程序
if __name__ == "__main__":
    build_database()