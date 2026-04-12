import os
import io
import base64
import pickle
from dotenv import load_dotenv
from PIL import Image

import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_debug

import embedding as emb

load_dotenv()
set_debug(True)

# ================= UI 配置 =================
st.set_page_config(page_title='MyLegal: 马来西亚智能法律顾问', page_icon="⚖️", layout='wide')

st.title('⚖️ MyLegal: 复合大模型法律智能体')
st.markdown("---")
st.sidebar.info("🎯 **架构展示**: \n1. 感知: Gemini Flash\n2. 主脑: Gemini Pro\n3. 检索: RAG + Rerank\n4. 推理: DeepSeek-R1")

uploaded_image = st.sidebar.file_uploader("📸 上传证据/合同截图 (可选)", type=["jpg", "png", "jpeg"])

def is_valid_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data))
        return True
    except:
        return False

# ================= 初始化模型军团 =================

vision_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

reasoning_model = ChatOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com", 
    model="deepseek-reasoner",
    temperature=0
)

@st.cache_resource
def load_agent():
    
    # --- 💡 核心防御：全新重写的工具定义 ---
    @tool
    def search_pdf_database(query: str) -> list:
        """检索马来西亚法律条文、法庭程序规范以及政府津贴指南数据库。"""
        
        print(f"\n[RAG] 开始检索: {query}")
        
        # 1. 粗排：获取可能包含字节码的原始文档
        try:
            base_docs = emb.base_retriever.invoke(query)
        except Exception as e:
            return[{"type": "text", "text": f"数据库检索失败: {e}"}]
        
        # 2. 绝对安全防御：现场暴力解冻
        unfrozen_docs =[]
        for doc in base_docs:
            # 修复点 1：直接判断 doc 本身是不是 bytes
            if isinstance(doc, bytes):
                try:
                    unwrapped_doc = pickle.loads(doc)
                    unfrozen_docs.append(unwrapped_doc)
                except Exception as e:
                    print(f"解冻 bytes 失败，跳过: {e}")
                    pass
            # 修复点 2：如果 doc 是正常对象，但它的 page_content 是 bytes
            elif hasattr(doc, 'page_content') and isinstance(doc.page_content, bytes):
                try:
                    unwrapped_doc = pickle.loads(doc.page_content)
                    unfrozen_docs.append(unwrapped_doc)
                except Exception as e:
                    print(f"解冻 page_content 失败，跳过: {e}")
                    pass
            else:
                unfrozen_docs.append(doc)
        
        if not unfrozen_docs:
            return [{"type": "text", "text": "检索完成，但未能成功解析法条数据。"}]

        # 3. 精排：使用 Reranker 提纯
        reranker = emb.compressor 
        try:
            reranked_docs = reranker.compress_documents(documents=unfrozen_docs, query=query)
        except Exception as e:
            print(f"[Rerank 警告] 精排失败，降级使用粗排结果: {e}")
            reranked_docs = unfrozen_docs
            
        # 4. 取最精华的前 5 条
        final_docs = reranked_docs[:5]
        
        # 5. 极简拼装（去除冗余的 parse_docs，直接提取文字和来源）
        text_list =[]
        for element in final_docs:
            # 安全提取文本
            text_val = element.page_content if hasattr(element, 'page_content') else str(element)
            
            # 修复点 3：安全提取 metadata 来源
            source_name = "本地法律库"
            if hasattr(element, 'metadata') and isinstance(element.metadata, dict):
                # 尝试拿 source_document，拿不到就拿 source，再拿不到就用默认值
                source_name = element.metadata.get('source_document', element.metadata.get('source', '本地法律库'))
                
            text_list.append(f"【来源: {source_name}】\n{text_val}")
                    
        context_text = '\n\n'.join(text_list)
        if not context_text:
            context_text = "未在本地数据库找到相关法律依据。"
            
        print(f"[RAG] 成功提取 {len(final_docs)} 条法条交由主脑分析。")
        return[{"type": "text", "text": f"检索到的法条：\n{context_text}"}]

    search_specific_websites = TavilySearchResults(
        max_results=3, 
        search_depth="advanced", 
        include_domains=["agc.gov.my", "mohr.gov.my", "malaysianbar.org.my", "hasil.gov.my","jtksm.mohr.gov.my"],
        description="当本地法律库找不到最新政策、办公地点时使用此工具搜索大马官方机构网站。" 
    )

    tools = [search_pdf_database, search_specific_websites]

    orchestrator_llm = ChatGoogleGenerativeAI(
        model='gemini-3.1-pro-preview',
        temperature=0, 
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    agent = create_agent(
        model=orchestrator_llm, 
        tools=tools, 
        system_prompt=(
            "你是一个大马法律资料搜查官 (Orchestrator)。"
            "你的任务是根据用户的问题，调用工具检索法律条文或上网搜集地址。"
            "【注意】：你不需要做最终的法律推断！你只需将你搜集到的所有客观事实、法律条文原文、相关地址，"
            "整理成一份详细的《法律事实备忘录》(Fact Memo)，输出交给下一环节的资深大律师处理。"
        )
    )

    return agent

my_agent = load_agent()

# ================= 对话逻辑与状态管理 =================

if "messages" not in st.session_state:
    st.session_state.messages =[]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("输入你的问题..."):
    
    display_prompt = prompt
    enhanced_prompt = prompt 
    
    st.session_state.messages.append({"role": "user", "content": display_prompt})
    with st.chat_message("user"):
        st.markdown(display_prompt)
        
    with st.chat_message("assistant"):
        
        # --- 阶段一：感知层 ---
        if uploaded_image is not None:
            with st.spinner("👀 视觉提取中..."):
                img_bytes = uploaded_image.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                vision_msg = HumanMessage(content=[
                    {"type": "text", "text": "提取图片中的所有文本事实，特别是涉及金额、日期、条款的内容。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ])
                vision_res = vision_model.invoke([vision_msg])
                extracted_facts = vision_res.content
                
                enhanced_prompt = f"用户原始问题: {prompt}\n\n[附件图片提取的事实]:\n{extracted_facts}"
                st.info("✅ 成功提取图片证据。")

        # --- 阶段二 & 三：主脑路由与检索 ---
        with st.spinner("🧠 主脑正在调度法律检索..."):
            
            temp_messages = st.session_state.messages[:-1] +[{"role": "user", "content": enhanced_prompt}]
            result = my_agent.invoke({"messages": temp_messages})
            raw_content = result["messages"][-1].content
            
            if isinstance(raw_content, list):
                investigation_report = "\n".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
            else:
                investigation_report = str(raw_content)
                
            investigation_report = investigation_report.replace("\\n", "\n")
            if "', 'extras': {" in investigation_report:
                investigation_report = investigation_report.split("', 'extras': {")[0]
                if investigation_report.startswith("('"):
                    investigation_report = investigation_report[2:]
            
            st.success("✅ 法律事实调查完毕。")
            with st.expander("查看 Agent 检索到的原始法条报告"):
                st.write(investigation_report)

        # --- 阶段四：深度推理与生成 ---
        with st.spinner("⚖️ 大律师正在进行深度逻辑推演..."):
            
            final_reasoning_prompt = f"""
            你是 MyLegal 资深大马律师。
            【用户的问题】: {enhanced_prompt}
            【调查员搜集到的法条与事实】: 
            {investigation_report}
            
            【你的任务】:
            请基于上述法律事实，对用户的问题进行严密的逻辑推演。
            1. 必须引用法条出处。
            2. 提供清晰的解决方案或步骤。
            3. 如果需要，起草法律信件。
            4. 请使用大马人容易理解的语气（可适当夹杂专业合规的 Manglish）。
            """
            
            r1_response = reasoning_model.invoke(final_reasoning_prompt)
            final_answer = r1_response.content
            st.markdown(final_answer)
            
    st.session_state.messages.append({"role": "assistant", "content": final_answer})