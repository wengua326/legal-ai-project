import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from tqdm import tqdm  

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
    temperature=0.1, 
    max_retries=10,  
)
output_parser = StrOutputParser()
summary_chain = LEGAL_SUMMARY_PROMPT | model | output_parser

def generate_summaries(chunks_list):
    print(f"\n🚀 [Summary 模块] 接收到 {len(chunks_list)} 个文本块。")
    if not chunks_list:
        return []

    texts_to_summarize = [chunk.page_content for chunk in chunks_list]
    all_summaries =[]
    
    
    batch_size = 50 
    
    print("Start Summarising...")
    
    
    for i in tqdm(range(0, len(texts_to_summarize), batch_size), desc="生成摘要进度"):
        
        
        current_batch_texts = texts_to_summarize[i : i + batch_size]
        
        try:
            
            batch_results = summary_chain.batch(
                current_batch_texts, 
                {"max_concurrency": 5} 
            )
            all_summaries.extend(batch_results)
            
        except Exception as e:
            print(f"\n❌ 处理批次 {i} 到 {i+batch_size} 时出错: {e}")
            
            all_summaries.extend(["[Error Summary]"] * len(current_batch_texts))

    print(f"\n🎉 [Summary 模块] 成功生成了 {len(all_summaries)} 条摘要！")
    return all_summaries
