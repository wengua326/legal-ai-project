# ⚖️ MyLegal AI: Integrating Malaysian Law with Neural Intelligence


An end-to-end, multi-agent legal assistant built for the **2026 National AI Competition (Innovation Track)**. 
MyLegal empowers Malaysian gig workers and MSMEs by democratizing access to justice, perfectly aligning with **UN SDG 8 (Decent Work & Economic Growth)**.

## 🚀 Live Demo & Links
- **Web App (Vercel):** [https://mylegal-assistant.vercel.app/]
- **Demo Video (YouTube):** [Watch our 2-min Pitch]
- **Backend API (Hugging Face):** [https://huggingface.co/spaces/wenguan326/mylegal-api]
- **Frontend Repo (Vercel):** []


---

## 🏗️ The Architecture: A 4-Tier Composite Agent

Our backend is not a simple LLM wrapper. It utilizes a state-of-the-art composite architecture to ensure legal accuracy, mitigate hallucinations, and provide actionable outcomes.

1. **Perception Layer (Gemini 2.5 Flash):** Multi-modal OCR engine capable of extracting critical facts from physical contracts or WhatsApp evidence.
2. **Orchestrator Layer (Gemini 3.1 Pro):** An intelligent router that dynamically assigns tools and enforces System Prompts based on the user's selected functional channel (e.g., Education, Labour, Contract Audit).
3. **Retrieval Layer (ChromaDB + Cohere Rerank):** A Two-Stage RAG (Retrieval-Augmented Generation) pipeline. It first retrieves relevant statutes from an index of over **800+ Malaysian Federal Acts and Landmark Case Laws**, then reranks them multilingually for maximum precision.
4. **Reasoning Layer (DeepSeek R1):** The core legal brain. It performs deep logical deduction strictly grounded in the retrieved facts to formulate legal strategies and draft formal documents.

---

## 🌟 The 5 Pillars of MyLegal

- 📚 **Legal Education:** Decodes Malaysian law using plain "Rojak" language.
- ✊ **Labour Rights Advocate:** Analyzes disputes (e.g., unfair dismissal) and provides step-by-step guidance and official JTK office locations.
- 👁️ **Smart Contract Auditor:** Scans uploaded PDFs or images for predatory clauses.
- ✍️ **Auto-Drafter:** Generates formatted, legally-sound Letters of Demand (LOD) via internal templates.
- 🧮 **Statutory Calculator:** Uses hardcoded Python logic to calculate exact statutory compensation (e.g., termination benefits, stamp duty) with zero math hallucination.

---

## ☁️ Deployment Strategy Note

To maintain a clean and professional codebase, the compiled vector databases (`chroma_db.zip`, `docstore.zip` - approx. 1GB) are **excluded** from this GitHub repository. 

We engineered a specialized `Dockerfile` that injects these massive document fragments into the container memory upon boot in our Hugging Face Space, bypassing traditional Git LFS bottlenecks and ensuring blazing-fast runtime retrieval.

---

## 👨‍💻 The Architects of Justice (Team)
- **[YONG WEN GUAN]** - AI Backend Architect & Team Lead
- **[KHAW JIA LE]** - Legal Data Engineer & Analyst
- **[ELVIS HENG ZI YOU]** - Frontend Developer & UI/UX Designer
- **[KHOR YI WEI]** - Product Manager & Legal Strategy
