import os
import io
import base64
import pickle
import json
from typing import List, Optional
from dotenv import load_dotenv
from PIL import Image
from contextlib import asynccontextmanager 

from pypdf import PdfReader 

# --- FastAPI Core Components ---
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

# --- LangChain Core Components ---
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.globals import set_debug

import embedding as emb
from calculate import SolicitorsRemunerationCalculator, TenancyStampDutyCalculator, EmploymentTerminationCalculator

load_dotenv()
set_debug(True)

# ================= Optimization 2: Eliminate frequent Disk I/O (In-memory Caching) =================
TEMPLATE_CACHE = {}
GUIDELINES_CACHE = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Lifespan] Loading legal templates and guidelines into memory cache...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    try:
        guideline_path = os.path.join(base_dir, 'lod_guidelines.json')
        with open(guideline_path, 'r', encoding='utf-8') as f:
            GUIDELINES_CACHE.update(json.load(f))
        print("✅ [Lifespan] Guidelines loaded successfully.")
    except Exception as e:
        print(f"❌ [Lifespan] Warning: Failed to load lod_guidelines.json: {e}")

    template_files = {'money': 'money.txt', 'defamation': 'defamation.txt', 'contract': 'contract.txt'}
    for key, filename in template_files.items():
        try:
            template_path = os.path.join(base_dir, 'templates', filename)
            with open(template_path, 'r', encoding='utf-8') as f:
                TEMPLATE_CACHE[key] = f.read()
            print(f"✅ [Lifespan] Template '{key}' loaded successfully.")
        except Exception as e:
            print(f"❌ [Lifespan] Warning: Failed to load template {filename}: {e}")
            
    yield 
    
    print("🛑 [Lifespan] Server is shutting down, clearing resources...")
    TEMPLATE_CACHE.clear()
    GUIDELINES_CACHE.clear()

# ================= 1. Initialize FastAPI =================
app = FastAPI(title="MyLegal API", description="Malaysian Legal AI Agent Backend", lifespan=lifespan)

#  New: Enable GZip compression to handle massive Base64 strings
app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 2. Define Data Models (Pydantic) =================
class ChatMessage(BaseModel):
    role: str
    content: str

class LegalRequest(BaseModel):
    prompt: str
    history: List[ChatMessage] = []
    file_base64: Optional[str] = None  
    file_type: Optional[str] = None 

# ================= 3. Model Initialization =================
vision_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash', temperature=0)

reasoning_model = ChatOpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"), 
    base_url="https://api.deepseek.com", 
    model="deepseek-reasoner",
    temperature=0
)

# ================= 4. Tools Definition =================

@tool
def search_pdf_database(query: str) -> list:
    """Retrieve data from the Malaysian legal statutes, case laws, and court procedures database. This tool must be used first for any legal inquiries!"""
    try:
        base_docs = emb.base_retriever.invoke(query)
    except Exception as e:
        return [{"type": "text", "text": f"Database retrieval failed: {str(e)}"}]
    
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
        return [{"type": "text", "text": "Retrieval completed, but failed to parse statutory data."}]
    
    try:
        reranker = emb.compressor 
        reranked_docs = reranker.compress_documents(documents=unfrozen_docs, query=query)
    except Exception as e:
        print(f"[Rerank Warning] Fine-ranking failed: {e}")
        reranked_docs = unfrozen_docs
        
    final_docs = reranked_docs[:5]
    text_list = []
    for element in final_docs:
        text_val = element.page_content if hasattr(element, 'page_content') else str(element)
        
        source_name = "Local Legal Database"
        if hasattr(element, 'metadata') and isinstance(element.metadata, dict):
            source_name = element.metadata.get('source_document', element.metadata.get('source', 'Local Legal Database'))
            
        text_list.append(f"[Source: {source_name}]\n{text_val}")
                
    context_text = '\n\n'.join(text_list) if text_list else "No relevant legal basis found in the local database."
    return [{"type": "text", "text": f"Retrieved Legal Provisions:\n{context_text}"}]

@tool
def search_malaysia_internet(query: str) -> str:
    """Use this tool to search official Malaysian government websites for the latest policies, specific addresses, or official announcements ONLY when the local database lacks such information."""
    official_domains = [
        "https://agc.gov.my", "https://mohr.gov.my", "https://malaysianbar.org.my",
        "https://hasil.gov.my", "https://kehakiman.gov.my", "https://smeinfo.com.my", 
        "https://mdec.my", "https://jtksm.mohr.gov.my"
    ]
    search = TavilySearchResults(max_results=3, search_depth="advanced", include_domains=official_domains)
    return search.invoke(f"{query} Malaysia")

@tool
def legal_fee_calculator(task_type: str, amount: float, extra_param: float = 0.0) -> str:
    """Dedicated calculator for Malaysian statutory fees. Absolutely precise; manual estimation is prohibited. task_type must be: 'lawyer_fee_A' (conveyancing), 'stamp_duty' (tenancy agreement), or 'termination' (severance pay)."""
    try:
        if task_type == 'lawyer_fee_A':
            fee = SolicitorsRemunerationCalculator.calculate_table_a(amount)
            return f"[Calculation Result] Standard legal fee for property valued at RM{amount:,.2f}: RM{fee:,.2f}"
        elif task_type == 'stamp_duty':
            duty = TenancyStampDutyCalculator.calculate(amount, extra_param)
            return f"[Calculation Result] Stamp duty for monthly rent RM{amount:,.2f} over a {extra_param}-year tenancy: RM{duty:,.2f}"
        elif task_type == 'termination':
            benefit = EmploymentTerminationCalculator.calculate(amount, extra_param)
            return f"[Calculation Result] Termination benefit for monthly salary RM{amount:,.2f} and {extra_param} years of service: RM{benefit:,.2f}"
        return "Unknown task type."
    except Exception as e:
        return f"Calculation Error: {str(e)}"

@tool
def fetch_legal_template(template_type: str) -> str:
    """Invoke this tool when the user requests drafting a formal letter (e.g., LOD). template_type must be: 'money' (debt recovery/unpaid wages), 'defamation' (cease and desist), or 'contract' (breach of contract)."""
    # 1. Retrieve template from memory cache
    template_content = TEMPLATE_CACHE.get(template_type)
    if not template_content:
        return f"System could not find a template for '{template_type}'. Available in cache: {list(TEMPLATE_CACHE.keys())}"
    
    # 2. Retrieve guidelines from memory cache
    if not GUIDELINES_CACHE:
        return "System could not find guideline configurations (GUIDELINES_CACHE is empty)."
        
    try:
        constraints = "\n- ".join(GUIDELINES_CACHE['system_instructions']['absolute_constraints'])
        blueprint = "\n- ".join([f"Step {item['step']} ({item['element']}): {item['requirements']}" for item in GUIDELINES_CACHE['system_instructions']['lod_blueprint']])

        return f"[Drafting Requirements (Must Comply)]:\n{constraints}\n[Structural Blueprint]:\n{blueprint}\n[Reference Template]:\n---\n{template_content}\n---"
    except Exception as e:
        return f"Failed to assemble template, please check JSON structure: {str(e)}"

# ================= 5. Core Refactoring: Dynamic Agent Routing Factory =================

def get_dynamic_agent(mode: str):
    """Dynamically mount/unmount tools and assign the strongest Prompt based on frontend routing signals."""
    orchestrator_llm = ChatGoogleGenerativeAI(
        model='gemini-3.1-pro-preview', temperature=0, 
        safety_settings={HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
    )

    
    if mode == "education":
        active_tools = [search_pdf_database]
        sys_prompt = "You are a Malaysian Legal Education Guide. You MUST ONLY use 'search_pdf_database' to retrieve statutes. Drafting documents, using the calculator, or browsing the internet is STRICTLY PROHIBITED. Extract objective legal facts and output a detailed Fact Memo in ENGLISH."
        
    elif mode == "labour":
        active_tools = [search_pdf_database, search_malaysia_internet]
        sys_prompt = "You are a Labour Rights Advocate. You MUST first use 'search_pdf_database' for statutes. If addresses for the Labour Department or courts are needed, use 'search_malaysia_internet'. Compile a Fact Memo in ENGLISH containing legal basis and action addresses."
        
    elif mode == "contract":
        active_tools = [search_pdf_database]
        sys_prompt = "You are a Contract Auditor. Rigorously cross-reference the user-provided contract text with statutes from 'search_pdf_database' to identify predatory or unfair clauses. Compile a Fact Memo in ENGLISH."
        
    elif mode == "lod":
        active_tools = [search_pdf_database, fetch_legal_template]
        sys_prompt = "You are a Legal Drafting Assistant. You MUST call 'fetch_legal_template' to obtain the drafting template, and verify the legal basis using 'search_pdf_database'. Compile the template and statutes into a Fact Memo in ENGLISH for the Senior Lawyer. If the user asks non-drafting questions, decline."
        
    elif mode == "calculator":
        active_tools = [legal_fee_calculator, search_pdf_database]
        sys_prompt = "You are a Statutory Fee Calculator. You are ABSOLUTELY PROHIBITED from estimating amounts manually! You must extract figures from the user input and invoke 'legal_fee_calculator'. Compile the results into a Fact Memo in ENGLISH. Decline non-calculation inquiries."
        
    else:
        active_tools = [search_pdf_database, search_malaysia_internet, legal_fee_calculator, fetch_legal_template]
        sys_prompt = "You are a versatile Malaysian Legal Orchestrator. Select appropriate tools based on the user's query and compile all objective facts into a detailed Fact Memo in ENGLISH for the Senior Lawyer to process."

    return create_agent(
        model=orchestrator_llm, 
        tools=active_tools, 
        system_prompt=sys_prompt
    )

# ================= 6. Core API Routing =================

@app.post("/api/chat")
def chat_handler(req: LegalRequest):
    print(f"\n[🚨 Request Received from Frontend] --->\n{req.prompt}\n<--- [Request End]\n")
    try:
        current_prompt = req.prompt
        
        current_mode = "default"
        clean_prompt = current_prompt
        
        
        if "Legal Education Mode" in current_prompt:
            current_mode = "education"
        elif "Labour Rights Mode" in current_prompt:
            current_mode = "labour"
        elif "Contract Audit Mode" in current_prompt:
            current_mode = "contract"
        elif "Document Drafting Mode" in current_prompt:
            current_mode = "lod"
        elif "Fee Calculation Mode" in current_prompt:
            current_mode = "calculator"

        print(f"🎯 Frontend Channel Routing Detected: [{current_mode.upper()}]")
        
        my_agent = get_dynamic_agent(current_mode)
        attachment_text = ""
        
        # --- Stage 1: Perception Layer (Image / PDF Dual Engine) ---
        if req.file_base64:
            pure_base64 = req.file_base64.split(",")[-1]
            pure_base64 += "=" * ((4 - len(pure_base64) % 4) % 4)
            file_mime_type = req.file_type.lower() if req.file_type else ""

            if "pdf" in file_mime_type:
                print(" PDF file uploaded. Initiating rapid local parsing...")
                try:
                    pdf_bytes = base64.b64decode(pure_base64)
                    pdf_file = io.BytesIO(pdf_bytes)
                    reader = PdfReader(pdf_file)
                    
                    pdf_text = ""
                    num_pages = min(10, len(reader.pages))
                    for i in range(num_pages):
                        extracted = reader.pages[i].extract_text()
                        if extracted:
                            pdf_text += extracted + "\n"
                            
                    attachment_text = f"[PDF Text Extracted Successfully (First {num_pages} pages)]:\n{pdf_text}"
                    print("✅ PDF text extraction complete.")
                except Exception as e:
                    print(f"❌ PDF reading failed: {e}")
                    attachment_text = "[Failed to read file, it may be corrupted or encrypted]"
            
            elif "image" in file_mime_type or file_mime_type == "":
                print("📸 Image uploaded. Calling Gemini Vision Model...")
                try:
                    vision_msg = HumanMessage(content=[
                        {"type": "text", "text": "Extract all factual text from the image, especially regarding amounts, dates, and terms."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pure_base64}"}}
                    ])
                    vision_res = vision_model.invoke([vision_msg])
                    attachment_text = f"[Image Text Extracted Successfully]:\n{vision_res.content}"
                    print("✅ Image evidence extraction complete.")
                except Exception as e:
                    print(f"❌ Gemini Vision parsing failed: {e}")
                    attachment_text = "[Failed to read image, please retry]"

            # Inject extracted content into the prompt
            clean_prompt = f"User's Original Query: {clean_prompt}\n\n[Content of Uploaded Attachment]:\n{attachment_text}"

        # --- Stage 2 & 3: Orchestrator Routing & Retrieval ---
        history_msgs = [{"role": m.role, "content": m.content} for m in req.history][-6:] 
        temp_messages = history_msgs + [{"role": "user", "content": clean_prompt}]
        
        agent_result = my_agent.invoke({"messages": temp_messages})
        raw_content = agent_result["messages"][-1].content
        
        if isinstance(raw_content, list):
            investigation_report = "\n".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
        else:
            investigation_report = str(raw_content)
            
        investigation_report = investigation_report.replace("\\n", "\n").split("', 'extras': {")[0]
        if investigation_report.startswith("('"): investigation_report = investigation_report[2:]

        # --- Stage 4: Deep Reasoning & Generation (DeepSeek R1) ---
        
        
        final_reasoning_prompt = f"""
        You are a Senior Malaysian Legal Counsel for MyLegal.
        CURRENT OPERATIONAL MODE: {current_mode.upper()}
        
        [USER'S QUERY]: 
        {clean_prompt}
        """
        
        if attachment_text:
            final_reasoning_prompt += f"\n[USER'S UPLOADED ATTACHMENT/EVIDENCE]: \n{attachment_text}\n\n"
            
        final_reasoning_prompt += f"""
        [FACTS & STATUTES GATHERED BY ORCHESTRATOR]: \n{investigation_report}
        
        [STRICT TASKS & FORMATTING RULES]:
        1. LANGUAGE ENFORCEMENT: You MUST respond in the EXACT SAME language the user used in their query (e.g., if the user asks in English, reply in English; if in Malay, reply in Bahasa Malaysia).
        2. LOD DRAFTING EXCEPTION: If in Document Drafting Mode (LOD), the drafted letter MUST ALWAYS be in formal English or Bahasa Malaysia, regardless of the user's input language.
        3. NO CONVERSATIONAL FILLERS: DO NOT output phrases like "Certainly," "Here is the letter," "Understood," or any greeting/closing remarks. Your output must begin IMMEDIATELY with the requested legal content or document draft.
        4. If in Education Mode, conclude with: "This content is for educational purposes only and does not constitute professional legal advice."
        5. If in Labour Rights Mode, provide clear actionable steps and append relevant institutional addresses based on the facts gathered.
        6. If in Contract Audit Mode, directly analyze the clauses in the [User's Uploaded Attachment/Evidence] against the law to identify predatory terms.
        7. Base your deductions strictly on facts and cite relevant statutory sources.
        """
        
        r1_response = reasoning_model.invoke(final_reasoning_prompt)
        
        return {
            "status": "success",
            "fact_memo": investigation_report, 
            "answer": r1_response.content
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        limit_concurrency=1000, 
        limit_max_requests=10000,
        h11_max_incomplete_event_size=52428800
    )
