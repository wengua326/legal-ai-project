import os
from dotenv import load_dotenv

# 1. 向量存储与文档库
from langchain_community.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# 2. 🌟 新增：Reranker 的魔法组件 🌟
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 加载环境变量
load_dotenv()

print("⚙️ 正在初始化 [双阶段] 法律检索系统...")

# ==========================================
# 阶段一：粗排 (Base Retriever) - 捞出前 20 本书
# ==========================================

# 1. 初始化 Embedding 模型 (保留你测试成功的模型)
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-2-preview" 
)

# 2. 连接 Chroma 数据库
VECTORSTORE_DIR = './chroma_db'
vectorstore = Chroma(
    collection_name='mylegal_rag', 
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_DIR
)

# 3. 连接本地文档库
DOCSTORE_DIR = './docstore'
store = LocalFileStore(DOCSTORE_DIR)
id_key = 'doc_id'

# 4. 配置底层 MultiVectorRetriever
# 🚨 关键修改：为了给 Reranker 留足“好苗子”，粗排阶段捞取数量 (k) 必须加大！
base_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    search_kwargs={
        'k': 20  # 以前是 8，现在我们先粗暴地捞出最相关的 20 条
    }
)

# ==========================================
# 阶段二：精排 (Reranker) - 选出最精华的 5 本书
# ==========================================

print("🧠 正在加载 Cohere 多语言重排序模型 (Multilingual Reranker)...")

# 1. 初始化 Cohere 压缩器 (指定多语言模型！)
# 确保你的 .env 里有 COHERE_API_KEY
cohere_api_key = os.environ.get("COHERE_API_KEY")
if not cohere_api_key:
    print("⚠️ 警告：未找到 COHERE_API_KEY。请检查 .env 文件。")

compressor = CohereRerank(
    model="rerank-multilingual-v3.0", # 🌟 核心：专治大马双语法条 🌟
    cohere_api_key=cohere_api_key,
    top_n=5 # 🌟 核心：从刚才的 20 条里，精挑细选出得分最高的 5 条交给大模型
)

# 2. 组装终极武器：将粗排和精排合并！
# 这个 final_retriever 才是我们以后在 Pipeline 里真正调用的神兵利器
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

print("✅ 双阶段检索系统初始化完毕！准备接受拷问。")

# (可选测试) 
if __name__ == "__main__":
    # 测试一下 Rojak 语检索
    test_query = "Boss pecat saya tak bagi notis, apa hak saya?"
    try:
         print(f"🔍 正在检索问题: '{test_query}'...")
         # 注意：调用的是 final_retriever！
         results = final_retriever.invoke(test_query)
         print(f"🎯 精排成功！找到了最精华的 {len(results)} 条法条。")
         # 打印第一条看看是不是命中要害
         if results:
             print(f"🥇 Top 1 结果预览:\n{results[0].page_content[:200]}...")
    except Exception as e:
         print(f"❌ 检索失败，请检查数据库或 API Key: {e}")