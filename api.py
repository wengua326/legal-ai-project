import os
import io
import base64
import pickle
import json
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image
from contextlib import asynccontextmanager 

# 🌟 新增：用于在后端极速读取 PDF 文本
from pypdf import PdfReader 

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

# 导入你的自定义模块
import embedding as emb
from calculate import SolicitorsRemunerationCalculator, TenancyStampDutyCalculator, EmploymentTerminationCalculator

load_dotenv()
set_debug(True)

# ================= 🌟 优化 2：消除频繁磁盘 I/O (内存缓存化) =================
TEMPLATE_CACHE = {}
GUIDELINES_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Lifespan] 正在将法律模板和指南加载到内存缓存中...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        guideline_path = os.path.join(base_dir, 'lod_guidelines.json')
        with open(guideline_path, 'r', encoding='utf-8') as f:
            GUIDELINES_CACHE.update(json.load(f))
        print("✅ [Lifespan] 指南加载成功")
    except Exception as e:
        print(f"❌ [Lifespan] 警告：无法加载 lod_guidelines.json: {e}")

    template_files = {'money': 'money.txt', 'defamation': 'defamation.txt', 'contract': 'contract.txt'}
    for key, filename in template_files.items():
        try:
            template_path = os.path.join(base_dir, 'templates', filename)
            with open(template_path, 'r', encoding='utf-8') as f:
                TEMPLATE_CACHE[key] = f.read()
            print(f"✅ [Lifespan] 模板 '{key}' 加载成功")
        except Exception as e:
            print(f"❌ [Lifespan] 警告：无法加载模板 {filename}: {e}")
            
    yield 
    
    print("🛑 [Lifespan] 服务器正在关闭，清理资源...")
    TEMPLATE_CACHE.clear()
    GUIDELINES_CACHE.clear()

# ================= 1. 初始化 FastAPI =================
app = FastAPI(title="MyLegal API", description="马来西亚法律智能体后端", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
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
    # 🌟 核心修改：将原来的 image_base64 泛化为 file_base64，并增加 file_type 字段
    file_base64: Optional[str] = None  
    file_type: Optional[str] = None # 例如: "image/jpeg", "image/png", "application/pdf"

# ================= 3. 模型初始化 =================
vision_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

reasoning_model = ChatOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com", 
    model="deepseek-reasoner",
    temperature=0
)

# ================= 4. 工具定义区 =================

@tool
def search_pdf_database(query: str) -> list:
    """检索大马法律法典、判例和法庭规则。只要涉及法律问题，必须先查这里！"""
    try:
        base_docs = emb.base_retriever.invoke(query)
    except Exception as e:
        return [{"type": "text", "text": f"数据库检索失败: {str(e)}"}]
    
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
            
    if not unfrozen_docs:
        return [{"type": "text", "text": "检索完成，但未能成功解析法条数据。"}]
    
    try:
        reranker = emb.compressor 
        reranked_docs = reranker.compress_documents(documents=unfrozen_docs, query=query)
    except Exception as e:
        print(f"[Rerank 警告] 精排失败: {e}")
        reranked_docs = unfrozen_docs
        
    final_docs = reranked_docs[:5]
    text_list = []
    for element in final_docs:
        text_val = element.page_content if hasattr(element, 'page_content') else str(element)
        
        source_name = "本地法律库"
        if hasattr(element, 'metadata') and isinstance(element.metadata, dict):
            source_name = element.metadata.get('source_document', element.metadata.get('source', '本地法律库'))
            
        text_list.append(f"【来源: {source_name}】\n{text_val}")
                
    context_text = '\n\n'.join(text_list) if text_list else "未在本地数据库找到相关法律依据。"
    return [{"type": "text", "text": f"检索到的法条：\n{context_text}"}]

@tool
def search_malaysia_internet(query: str) -> str:
    """仅当本地库找不到信息时，通过大马政府官方网站查询最新政策、地址或公告。"""
    official_domains = [
        "https://agc.gov.my", "https://mohr.gov.my", "https://malaysianbar.org.my",
        "https://hasil.gov.my", "https://kehakiman.gov.my", "https://smeinfo.com.my", 
        "https://mdec.my", "https://jtksm.mohr.gov.my"
    ]
    search = TavilySearchResults(max_results=3, search_depth="advanced", include_domains=official_domains)
    return search.invoke(f"{query} Malaysia")

@tool
def legal_fee_calculator(task_type: str, amount: float, extra_param: float = 0.0) -> str:
    """专门用于计算大马法定费用。task_type: 'lawyer_fee_A', 'stamp_duty', 'termination'"""
    try:
        if task_type == 'lawyer_fee_A':
            fee = SolicitorsRemunerationCalculator.calculate_table_a(amount)
            return f"[计算结果] RM{amount:,.2f} 房产标准律师费: RM{fee:,.2f}"
        elif task_type == 'stamp_duty':
            duty = TenancyStampDutyCalculator.calculate(amount, extra_param)
            return f"[计算结果] 月租 RM{amount:,.2f}，租期 {extra_param} 年的印花税: RM{duty:,.2f}"
        elif task_type == 'termination':
            benefit = EmploymentTerminationCalculator.calculate(amount, extra_param)
            return f"[计算结果] 月薪 RM{amount:,.2f}，服务 {extra_param} 年的遣散费: RM{benefit:,.2f}"
        return "未知的任务类型。"
    except Exception as e:
        return f"计算错误: {str(e)}"

@tool
def fetch_legal_template(template_type: str) -> str:
    """当用户要求起草信件(LOD)时调用。template_type: 'money', 'defamation', 'contract'。"""
    file_map = {'money': 'money.txt', 'defamation': 'defamation.txt', 'contract': 'contract.txt'}
    target_file = file_map.get(template_type)
    if not target_file: return "系统未找到对应模板，请凭借大马法律知识自行起草。"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        guideline_path = os.path.join(base_dir, 'lod_guidelines.json')
        with open(guideline_path, 'r', encoding='utf-8') as f: guidelines_data = json.load(f)
        constraints = "\n- ".join(guidelines_data['system_instructions']['absolute_constraints'])
        blueprint = "\n- ".join([f"Step {item['step']} ({item['element']}): {item['requirements']}" for item in guidelines_data['system_instructions']['lod_blueprint']])
        
        template_path = os.path.join(base_dir, 'templates', target_file)
        with open(template_path, 'r', encoding='utf-8') as f: template_content = f.read()

        return f"【起草要求】:\n{constraints}\n【结构规范】:\n{blueprint}\n【参考模板】:\n---\n{template_content}\n---"
    except Exception as e:
        return f"读取文件失败: {str(e)}"


# ================= 5. 核心改造：动态 Agent 路由工厂 =================

def get_dynamic_agent(mode: str):
    """根据前端传来的暗号，动态卸载/挂载武器，并分配最强 Prompt"""
    orchestrator_llm = ChatGoogleGenerativeAI(
        model='gemini-3.1-pro-preview', temperature=0, 
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    if mode == "education":
        active_tools = [search_pdf_database]
        sys_prompt = "你是大马普法向导。你【只能】使用 search_pdf_database 检索法条。严禁起草文书，严禁使用计算器，严禁上网。只需提取客观法律事实输出 Fact Memo。"
        
    elif mode == "labour":
        active_tools = [search_pdf_database, search_malaysia_internet]
        sys_prompt = "你是劳工维权向导。你必须先用 search_pdf_database 查法条，如果需要找劳工部或法庭地址，调用 search_malaysia_internet。整理包含法条依据和维权地址的 Fact Memo。"
        
    elif mode == "contract":
        active_tools = [search_pdf_database]
        sys_prompt = "你是合同审计员。根据用户提供的合同内容，严密比对 search_pdf_database 中的法条，找出霸王条款。整理 Fact Memo。"
        
    elif mode == "lod":
        active_tools = [search_pdf_database, fetch_legal_template]
        sys_prompt = "你是文书起草助理。你【必须】调用 fetch_legal_template 获取模板，并用 search_pdf_database 核实法律依据。将模板和法条整理成 Fact Memo 交给大律师。如果用户提问无关律师信的事务请拒绝回答，并表示只能处理律师信事务。"
        
    elif mode == "calculator":
        active_tools = [legal_fee_calculator, search_pdf_database]
        sys_prompt = "你是法定费用核算员。你【绝对禁止】自己口算！必须从用户输入提取数字，调用 legal_fee_calculator。将计算结果整理成 Fact Memo。如果用户提问无关计算的事务请拒绝回答,并表示只能处理计算事务。"
        
    else:
        active_tools = [search_pdf_database, search_malaysia_internet, legal_fee_calculator, fetch_legal_template]
        sys_prompt = "你是一个全能的大马法律资料搜查官 (Orchestrator)。根据问题选择合适的工具，整理 Fact Memo 输出交给大律师处理。"

    return create_agent(
        model=orchestrator_llm, 
        tools=active_tools, 
        system_prompt=sys_prompt
    )

# ================= 6. 核心 API 路由 =================

@app.post("/api/chat")
def chat_handler(req: LegalRequest):
    print(f"\n[🚨 前端发来的请求] --->\n{req.prompt}\n<--- [请求结束]\n")
    try:
        current_prompt = req.prompt
        
        # 🕵️‍♂️ 拦截前端暗号，决定模式！
        current_mode = "default"
        clean_prompt = current_prompt
        
        if "普法教育模式" in current_prompt:
            current_mode = "education"
        elif "劳工维权模式" in current_prompt:
            current_mode = "labour"
        elif "合同审计模式" in current_prompt:
            current_mode = "contract"
        elif "文书起草模式" in current_prompt:
            current_mode = "lod"
        elif "费用计算模式" in current_prompt:
            current_mode = "calculator"

        print(f"🎯 侦测到前端频道路由: [{current_mode.upper()}]")
        
        my_agent = get_dynamic_agent(current_mode)
        enhanced_prompt = clean_prompt
        
        # --- 🌟 阶段一：感知层 (图片/PDF 双擎处理) 🌟 ---
        if req.file_base64:
            pure_base64 = req.file_base64.split(",")[-1]
            file_mime_type = req.file_type.lower() if req.file_type else ""
            extracted_facts = ""

            # 逻辑 A: 如果前端明确传过来的是 PDF 文件
            if "pdf" in file_mime_type:
                print("📄 检测到用户上传了 PDF 文件，正在本地极速解析...")
                try:
                    pdf_bytes = base64.b64decode(pure_base64)
                    pdf_file = io.BytesIO(pdf_bytes)
                    reader = PdfReader(pdf_file)
                    
                    pdf_text = ""
                    # 限制最多只读前 10 页，防止超大 PDF 撑爆内存
                    num_pages = min(10, len(reader.pages))
                    for i in range(num_pages):
                        extracted = reader.pages[i].extract_text()
                        if extracted:
                            pdf_text += extracted + "\n"
                            
                    extracted_facts = f"[PDF 文本提取成功 (截取前 {num_pages} 页)]:\n{pdf_text}"
                    print("✅ PDF 文本提取完成。")
                except Exception as e:
                    print(f"❌ PDF 读取失败: {e}")
                    extracted_facts = "[读取文件失败，可能文件已损坏或加密]"
            
            # 逻辑 B: 如果是图片，依然使用强大的 Gemini Vision 模型
            elif "image" in file_mime_type or file_mime_type == "":
                print("📸 检测到用户上传了图片，正在呼叫 Gemini Vision 模型...")
                try:
                    vision_msg = HumanMessage(content=[
                        {"type": "text", "text": "提取图片中的所有文本事实，特别是涉及金额、日期、条款的内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pure_base64}"}}
                    ])
                    vision_res = vision_model.invoke([vision_msg])
                    extracted_facts = f"[图片文本提取成功]:\n{vision_res.content}"
                    print("✅ 图片证据提取完成。")
                except Exception as e:
                    print(f"❌ Gemini Vision 解析失败: {e}")
                    extracted_facts = "[读取图片失败，请重试]"

            # 将提取出的内容（无论是 PDF 还是图片）注入给后方的法律大脑
            enhanced_prompt = f"用户原始问题: {clean_prompt}\n\n[用户上传附件的内容]:\n{extracted_facts}"

        # --- 阶段二 & 三：主脑路由与检索 ---
        history_msgs = [{"role": m.role, "content": m.content} for m in req.history][-6:] 
        temp_messages = history_msgs +[{"role": "user", "content": enhanced_prompt}]
        
        agent_result = my_agent.invoke({"messages": temp_messages})
        raw_content = agent_result["messages"][-1].content
        
        if isinstance(raw_content, list):
            investigation_report = "\n".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
        else:
            investigation_report = str(raw_content)
            
        investigation_report = investigation_report.replace("\\n", "\n").split("', 'extras': {")[0]
        if investigation_report.startswith("('"): investigation_report = investigation_report[2:]

        # --- 阶段四：深度推理与生成 (DeepSeek R1) ---
        final_reasoning_prompt = f"""
        你是 MyLegal 资深大马律师。
        当前所处模式：{current_mode}
        【用户的问题】: {enhanced_prompt}
        【调查员搜集到的资料】: \n{investigation_report}
        
        【任务】:
        1. 如果是普法模式，必须声明"本内容仅供教育参考，不构成专业法律建议"。
        2. 如果是文书起草模式,必须根据调查员提供的【参考模板】和guidelines生成信件草稿,不要进行过度修改。
        3. 如果是维权模式，除了法律推演，还要给出清晰步骤（并附上机构地址）。
        4. 请基于事实严密推演并引用法条出处。用大马人易懂的口吻回答。
        """
        
        r1_response = reasoning_model.invoke(final_reasoning_prompt)
        
        return {
            "status": "success",
            "fact_memo": investigation_report, 
            "answer": r1_response.content
        }

    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)