import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
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
model = ChatGroq(
    model='llama-3.1-8b-instant', 
    temperature=0.1, # 降低温度，确保摘要稳定
    max_retries=10,  # 🌟 关键防御：遇到限流自动重试，防止直接崩溃
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
    batch_size = 50 # 每次送 500 个给 API
    
    print("⏳ 开始呼叫 Gemini API 进行批量总结 (预计需要 1-2 小时，请喝杯咖啡)...")
    
    # tqdm 会自动在终端生成一个非常漂亮的动态进度条 [██████████░░░] 80%
    for i in tqdm(range(0, len(texts_to_summarize), batch_size), desc="生成摘要进度"):
        
        # 切割出当前批次的 500 个文本
        current_batch_texts = texts_to_summarize[i : i + batch_size]
        
        try:
            # 这一批次内部，依然保持 15 的高并发
            batch_results = summary_chain.batch(
                current_batch_texts, 
                {"max_concurrency": 5} 
            )
            all_summaries.extend(batch_results)
            
        except Exception as e:
            print(f"\n❌ 处理批次 {i} 到 {i+batch_size} 时出错: {e}")
            # 如果出错，用空字符串占位，保证总数量不乱
            all_summaries.extend(["[Error Summary]"] * len(current_batch_texts))

    print(f"\n🎉 [Summary 模块] 成功生成了 {len(all_summaries)} 条摘要！")
    return all_summaries