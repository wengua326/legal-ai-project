

import os
from dotenv import load_dotenv

# Vector storage and document store
from langchain_community.vectorstores import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever

# Reranker core components
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank


load_dotenv()

print("Initializing legal retrieval system...")

# Stage 1: Initial Retrieval (Base Retriever) 
# ============================

# 1. Initialize Embedding Model 
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview" 
)

# 2. Connect to Chroma Database
VECTORSTORE_DIR = './chroma_db'
vectorstore = Chroma(
    collection_name='mylegal_rag', 
    embedding_function=embedding_model,
    persist_directory=VECTORSTORE_DIR
)

# 3. Connect to Local Document Store
DOCSTORE_DIR = './docstore'
store = LocalFileStore(DOCSTORE_DIR)
id_key = 'doc_id'

# 4. Configure underlying MultiVectorRetriever
base_retriever = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=store,
    id_key=id_key,
    search_kwargs={
        'k': 20 
    }
)


# Stage 2: Reranking (Reranker) 
# ======================

print("Loading Cohere Reranker...")

# Initialize Cohere Compressor 
cohere_api_key = os.environ.get("COHERE_API_KEY")
if not cohere_api_key:
    print("⚠️ Warning: COHERE_API_KEY not found. Please check your .env file.")

compressor = CohereRerank(
    model="rerank-v4.0-pro", 
    cohere_api_key=cohere_api_key,
    top_n=5 
)

# Combine initial retrieval and reranking
final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

print("Two-stage retrieval system initialized successfully!")
