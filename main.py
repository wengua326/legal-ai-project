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
from langchain_openai import ChatOpenAI # 💡 新增：用于调用 DeepSeek R1
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_debug

# 导入你优化后的底层检索引擎 (Tier 3: Gemini Embedding 2 + Cohere Reranker)
import embedding as emb

# 加载环境变量 (需要新增 DEEPSEEK_API_KEY)
load_dotenv()
set_debug(True)

# ================= UI 配置 =================
st.set_page_config(page_title='MyLegal: 马来西亚智能法律顾问', page_icon="⚖️", layout='wide')

st.title('⚖️ MyLegal: 复合大模型法律智能体')
st.markdown("---")
st.sidebar.info("🎯 **架构展示**: \n1. 感知: Gemini Flash\n2. 主脑: Gemini Pro\n3. 检索: RAG + Rerank\n4. 推理: DeepSeek-R1")

#  新增：侧边栏图片上传 (Tier 1: 视觉感知入口)
uploaded_image = st.sidebar.file_uploader("📸 上传证据/合同截图 (可选)", type=["jpg", "png", "jpeg"])

def is_valid_image(data: bytes) -> bool:
    try:
        Image.open(io.BytesIO(data))
        return True
    except:
        return False

# ================= 初始化模型军团 =================

# 💡 Tier 1: 视觉提取模型 (Gemini 2.5 Flash)
vision_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', temperature=0) # 注: API名称以1.5为主，实际调用最新flash能力

# 💡 Tier 4: 深度推理模型 (DeepSeek R1)
# DeepSeek 的 API 完全兼容 OpenAI 格式
reasoning_model = ChatOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", "YOUR_DEEPSEEK_KEY_HERE"), 
    base_url="https://api.deepseek.com", 
    model="deepseek-reasoner", # R1 模型
    temperature=0
)

@st.cache_resource
def load_agent():
    # --- 保持原始核心解析逻辑不动 ---
    def parse_docs(docs):
        b64_images = []
        texts =[]
        for doc in docs:
            content = doc.page_content if hasattr(doc, 'page_content') else doc
            if isinstance(content, bytes):
                try:
                    unwrapped_doc = pickle.loads(content)
                    texts.append(unwrapped_doc)
                    continue
                except Exception:
                    pass
                try:
                    content_str = content.decode('utf-8')
                    if "iVB0" in content_str[:50] or "/9j/" in content_str[:50]:
                        b64_images.append(content_str)
                        continue
                except Exception:
                    pass
            elif isinstance(content, str):
                if "iVB0" in content[:50] or "/9j/" in content[:50]:
                    b64_images.append(content)
                else:
                    texts.append(content)
        return {'images': b64_images, 'texts': texts}

    # --- 保持原始工具定义不动 ---
    @tool
    def search_pdf_database(query: str) -> list:
        """检索马来西亚法律条文、法庭程序规范以及政府津贴指南数据库。"""
        try:
            docs = emb.final_retriever.invoke(query) 
        except AttributeError:
            docs = emb.retriever.invoke(query) 
        
        docs = docs[:5] 
        parsed_data = parse_docs(docs)
        
        content_blocks =[]
        text_list = []
        
        if len(parsed_data['texts']) > 0:
            for element in parsed_data['texts']:
                element_type = str(type(element)).lower()
                if 'table' in element_type:
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'text_as_html') and element.metadata.text_as_html:
                        text_list.append(f"\n--- [表格数据] ---\n{element.metadata.text_as_html}\n")
                    else:
                        text_list.append(f"\n--- [表格数据] ---\n{element.text}\n")
                else:
                    source_name = getattr(element, 'metadata', {}).get('source_document', '本地法律库')
                    text_val = element.text if hasattr(element, 'text') else str(element)
                    text_list.append(f"【来源: {source_name}】\n{text_val}")
                    
        context_text = '\n\n'.join(text_list)
        if not context_text:
            context_text = "未在本地数据库找到相关法律依据。"
            
        content_blocks.append({"type": "text", "text": f"检索到的法条：\n{context_text}"})
        return content_blocks

    search_specific_websites = TavilySearchResults(
        max_results=3, 
        search_depth="advanced", 
        include_domains=["agc.gov.my", "mohr.gov.my", "malaysianbar.org.my", "hasil.gov.my"],
        description="当本地法律库找不到最新政策、办公地点时使用此工具搜索大马官方机构网站。" 
    )

    tools = [search_pdf_database, search_specific_websites]

    # 💡 Tier 2: 主脑指挥官 (Gemini 3.1 Pro / 1.5 Pro)
    orchestrator_llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-pro-latest', # 主脑用 Pro，参数量大，工具调用极稳
        temperature=0, 
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    # 💡 核心优化：修改 Agent 的职责！它不再直接回答用户，而是生成“调查报告”
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
    
    # 记录用户的原始问题
    display_prompt = prompt
    enhanced_prompt = prompt # 这是最终发给后台的 prompt
    
    st.session_state.messages.append({"role": "user", "content": display_prompt})
    with st.chat_message("user"):
        st.markdown(display_prompt)
        
    with st.chat_message("assistant"):
        
        # ---------------------------------------------------------
        # 🟢 阶段一：感知层 (Vision Processing)
        # ---------------------------------------------------------
        if uploaded_image is not None:
            with st.spinner("👀 视觉提取中 (Gemini Flash)..."):
                # 把图片转为 Base64 给大模型看
                img_bytes = uploaded_image.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                
                vision_msg = HumanMessage(content=[
                    {"type": "text", "text": "提取图片中的所有文本事实，特别是涉及金额、日期、条款的内容。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ])
                vision_res = vision_model.invoke([vision_msg])
                extracted_facts = vision_res.content
                
                # 将图片内容补充到用户的提问中
                enhanced_prompt = f"用户原始问题: {prompt}\n\n[附件图片提取的事实]:\n{extracted_facts}"
                st.info("✅ 成功提取图片证据。")

        # ---------------------------------------------------------
        # 🔵 阶段二 & 三：主脑路由与检索 (Orchestrator & RAG)
        # ---------------------------------------------------------
        with st.spinner("🧠 主脑正在调度法律检索 (Gemini Pro)..."):
            
            # 我们给 Agent 发送增强后的 Prompt
            temp_messages = st.session_state.messages[:-1] + [{"role": "user", "content": enhanced_prompt}]
            
            result = my_agent.invoke({"messages": temp_messages})
            raw_content = result["messages"][-1].content
            
            # --- 保持你原始的清洗逻辑不动 ---
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

        # ---------------------------------------------------------
        # 🔴 阶段四：深度推理与生成 (DeepSeek R1)
        # ---------------------------------------------------------
        with st.spinner("⚖️ 大律师正在进行深度逻辑推演 (DeepSeek R1)..."):
            
            # 构建给 DeepSeek 的终极指令
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
            
            # 调用 DeepSeek R1
            r1_response = reasoning_model.invoke(final_reasoning_prompt)
            final_answer = r1_response.content
            
            # 显示最终答案
            st.markdown(final_answer)
            
    # 记录到对话历史中（只记录用户的原始问题和 R1 的最终回答）
    st.session_state.messages.append({"role": "assistant", "content": final_answer})