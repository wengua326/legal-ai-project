from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chunking as ck  # 【保留】我们依然需要从这里拿到切好的 chunks

# 1. 【改造一】只保留一个针对法律文本的 Prompt
# 这个 Prompt 强调提取关键词，为向量检索做准备
LEGAL_SUMMARY_PROMPT = ChatPromptTemplate.from_template(
    """You are a highly intelligent AI paralegal. 
    Your task is to create a concise, highly searchable summary of the following legal text chunk.
    
    Focus on extracting the key legal concepts, actions, entities (e.g., 'employer', 'employee'), and any specific conditions or penalties mentioned.
    This summary will be used for vector search, so it must be dense with keywords.
    
    DO NOT alter the legal meaning or add personal opinions. Respond only with the summary.

    Legal Text Chunk:
    ---
    {element}
    ---
    """
)

# 2. 初始化大模型和输出解析器
# 用 Llama 3.1 8B 就足够快且便宜，温度可以调低一点保证稳定性
model = ChatGroq(temperature=0.2, model='llama-3.1-8b-instant')
output_parser = StrOutputParser()

# 3. 构建我们的“总结链” (Summary Chain)
summary_chain = LEGAL_SUMMARY_PROMPT | model | output_parser

# 4. 【改造二】删除所有关于 Table 和 Image 的代码
# 我们只处理文本，所以只需要一个列表来存总结结果
text_summaries = []

def generate_summaries():
    """
    批量处理所有文本 chunks，生成摘要。
    """
    print("🚀 开始为所有文本块生成核心摘要...")
    
    # 从新的 chunking.py 拿到数据
    # 我们只关心 chunks 里的 page_content
    chunks_to_summarize = [chunk.page_content for chunk in ck.chunks]
    
    if not chunks_to_summarize:
        print("❌ 没有找到任何文本块来生成摘要。")
        return []

    # 【改造三】使用 .batch() 进行批量并行处理，极大提升速度
    try:
        summaries = summary_chain.batch(
            chunks_to_summarize, 
            {"max_concurrency": 5} # 【可调参数】同时向 API 发送 5 个请求
        )
        print(f"🎉 成功生成了 {len(summaries)} 条摘要！")
        return summaries
    except Exception as e:
        print(f"❌ 生成摘要时出错: {e}")
        return []

# 主程序：调用函数并将结果存入变量，供下一步的 build_db.py 使用
text_summaries = generate_summaries()

# (可选) 测试一下，看看第一条摘要长什么样
if __name__ == "__main__":
    if text_summaries:
        print("\n--- 第一条摘要预览 ---")
        print(text_summaries[0])
        print("\n--- 对应的原始文本块 ---")
        print(ck.chunks[0].page_content[:300] + "...")