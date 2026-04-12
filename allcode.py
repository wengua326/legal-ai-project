#summary
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm  # 🌟 引入进度条神器

load_dotenv()

LEGAL_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a highly intelligent AI paralegal. 
    Your task is to create a concise, highly searchable summary of the following legal text chunk.
    Focus on extracting key legal concepts, actions, entities, and penalties.
    DO NOT alter the legal meaning. Respond only with the summary.

    Legal Text Chunk:
    ---
    {element}
    ---
    """
)
model = ChatGoogleGenerativeAI(
    model='gemini-flash-lite-latest', # 速度与力量的平衡
    temperature=0,
    max_retries=10, 
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
)
output_parser = StrOutputParser()
summary_chain = LEGAL_SUMMARY_PROMPT | model | output_parser

def generate_summaries(chunks_list):
    print(f"\n🚀 [Summary 模块] 接收到 {len(chunks_list)} 个文本块。")
    if not chunks_list:
        return []

    texts_to_summarize = [chunk.page_content for chunk in chunks_list]
    all_summaries =[]
    
    # 🌟 核心改造：分批次送去总结，防止内存溢出，并显示进度条！
    batch_size = 500 # 每次送 500 个给 API
    
    print("⏳ 开始呼叫 Gemini API 进行批量总结 (预计需要 1-2 小时，请喝杯咖啡)...")
    
    # tqdm 会自动在终端生成一个非常漂亮的动态进度条 [██████████░░░] 80%
    for i in tqdm(range(0, len(texts_to_summarize), batch_size), desc="生成摘要进度"):
        
        # 切割出当前批次的 500 个文本
        current_batch_texts = texts_to_summarize[i : i + batch_size]
        
        try:
            # 这一批次内部，依然保持 15 的高并发
            batch_results = summary_chain.batch(
                current_batch_texts, 
                {"max_concurrency": 20} 
            )
            all_summaries.extend(batch_results)
            
        except Exception as e:
            print(f"\n❌ 处理批次 {i} 到 {i+batch_size} 时出错: {e}")
            # 如果出错，用空字符串占位，保证总数量不乱
            all_summaries.extend(["[Error Summary]"] * len(current_batch_texts))

    print(f"\n🎉 [Summary 模块] 成功生成了 {len(all_summaries)} 条摘要！")
    return all_summaries

#embedding
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
    model="rerank-v4.0-pro", # 🌟 核心：专治大马双语法条 🌟
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

#data_cleaning
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


#chunking
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

#build_db
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

    backup_file = "mylegal_backup.pkl"
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
    
    # 彻底删除旧的文件夹，防止合并冲突
    # 注意：如果你想分多次运行 build_db 追加数据，请注释掉下面这两行
    # os.system('rmdir /s /q chroma_db') 
    # os.system('rmdir /s /q docstore')

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

#api
import os
import io
import base64
import pickle
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image

# --- FastAPI 核心组件 ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- LangChain 核心组件 ---
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_debug

# 导入你优化后的检索引擎
import embedding as emb

load_dotenv()
set_debug(True)

# ================= 1. 初始化 FastAPI =================
app = FastAPI(title="MyLegal API", description="马来西亚法律智能体后端")

# 🚨 解决跨域 (CORS)，允许 Vercel 前端访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 生产环境建议改为具体的 Vercel 域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. 定义数据格式 (Pydantic) =================
class ChatMessage(BaseModel):
    role: str
    content: str

class LegalRequest(BaseModel):
    prompt: str
    history: List[ChatMessage] = []
    image_base64: Optional[str] = None  # 接收前端传来的图片 Base64

# ================= 3. 模型初始化 (保留原始逻辑) =================
vision_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

reasoning_model = ChatOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com", 
    model="deepseek-reasoner",
    temperature=0
)

# ================= 4. Agent 与 工具定义 (保留原始逻辑) =================

@tool
def search_pdf_database(query: str) -> list:
    """检索马来西亚法律条文、法庭程序规范以及政府津贴指南数据库。"""
    # 1. 粗排
    base_docs = emb.base_retriever.invoke(query)
    
    # 2. 现场解冻 (保留你的核心修复)
    unfrozen_docs = []
    for doc in base_docs:
        if isinstance(doc, bytes):
            try: unfrozen_docs.append(pickle.loads(doc))
            except: pass
        elif hasattr(doc, 'page_content') and isinstance(doc.page_content, bytes):
            try: unfrozen_docs.append(pickle.loads(doc.page_content))
            except: pass
        else:
            unfrozen_docs.append(doc)
    
    # 3. 精排
    reranker = emb.compressor 
    try:
        reranked_docs = reranker.compress_documents(documents=unfrozen_docs, query=query)
    except:
        reranked_docs = unfrozen_docs
        
    final_docs = reranked_docs[:5]
    
    # 4. 文本拼装
    text_list = []
    for element in final_docs:
        text_val = element.page_content if hasattr(element, 'page_content') else str(element)
        source_name = "本地法律库"
        if hasattr(element, 'metadata') and isinstance(element.metadata, dict):
            source_name = element.metadata.get('source_document', element.metadata.get('source', '本地法律库'))
        text_list.append(f"【来源: {source_name}】\n{text_val}")
                
    context_text = '\n\n'.join(text_list)
    return [{"type": "text", "text": f"检索到的法条：\n{context_text}"}]

search_specific_websites = TavilySearchResults(
    max_results=3, 
    search_depth="advanced", 
    include_domains=["agc.gov.my", "mohr.gov.my", "malaysianbar.org.my", "hasil.gov.my"],
    description="当本地法律库找不到信息时使用此工具搜索大马官方机构网站。" 
)

def get_my_agent():
    orchestrator_llm = ChatGoogleGenerativeAI(
        model='gemini-3.1-pro-preview',
        temperature=0, 
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )
    return create_agent(
        model=orchestrator_llm, 
        tools=[search_pdf_database, search_specific_websites], 
        system_prompt=(
            "你是一个大马法律资料搜查官 (Orchestrator)。"
            "你的任务是根据用户的问题，调用工具检索法律条文或上网搜集地址。"
            "你只需将你搜集到的所有客观事实整理成详细的《法律事实备忘录》(Fact Memo)，输出交给下一环节处理。"
        )
    )

my_agent = get_my_agent()

# ================= 5. 核心 API 路由 =================

@app.post("/api/chat")
async def chat_handler(req: LegalRequest):
    try:
        current_prompt = req.prompt
        enhanced_prompt = current_prompt
        
        # --- 阶段一：感知层 (视觉) ---
        if req.image_base64:
            # 去掉可能的 data:image/jpeg;base64, 前缀
            pure_base64 = req.image_base64.split(",")[-1]
            vision_msg = HumanMessage(content=[
                {"type": "text", "text": "提取图片中的所有文本事实，特别是涉及金额、日期、条款的内容。"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pure_base64}"}}
            ])
            vision_res = vision_model.invoke([vision_msg])
            enhanced_prompt = f"用户原始问题: {current_prompt}\n\n[附件图片提取的事实]:\n{vision_res.content}"

        # --- 阶段二 & 三：主脑路由与检索 ---
        # 转换历史记录格式
        history_msgs = [{"role": m.role, "content": m.content} for m in req.history]
        # 只取最近几轮防止 Token 爆炸
        history_msgs = history_msgs[-6:] 
        
        temp_messages = history_msgs + [{"role": "user", "content": enhanced_prompt}]
        
        agent_result = my_agent.invoke({"messages": temp_messages})
        raw_content = agent_result["messages"][-1].content
        
        # 清洗报告文本
        if isinstance(raw_content, list):
            investigation_report = "\n".join([item.get("text", "") for item in raw_content if isinstance(item, dict)])
        else:
            investigation_report = str(raw_content)
        
        # --- 阶段四：深度推理与生成 (DeepSeek R1) ---
        final_reasoning_prompt = f"""
        你是 MyLegal 资深大马律师。
        【用户的问题】: {enhanced_prompt}
        【调查员搜集到的法条与事实】: {investigation_report}
        请基于上述事实，引用法条出处，进行严密推演并提供解决方案。用大马人易懂的口吻回答。
        """
        
        r1_response = reasoning_model.invoke(final_reasoning_prompt)
        
        return {
            "status": "success",
            "fact_memo": investigation_report, # 返回调查报告，方便前端用展开栏显示
            "answer": r1_response.content
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ================= 启动 =================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)