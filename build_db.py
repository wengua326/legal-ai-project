
import os
import uuid
import pickle
from dotenv import load_dotenv
from langchain_core.documents import Document
from tqdm import tqdm

load_dotenv()

# Import core modules
import chunking as ck
import summary as sm
import embedding as emb

def build_database():
    print("MyLegal Database")
    print("================\n")

    backup_file = "mylegal_backup_update_v2.pkl"
    my_chunks = []
    my_summaries = []

    # Step 1: 
    if os.path.exists(backup_file):
        print(f"📂 Found local backup file '{backup_file}', loading...")
        try:
            with open(backup_file, "rb") as f:
                data = pickle.load(f)
                my_chunks = data['chunks']
                my_summaries = data['summaries']
            print(f"✅ Loaded successfully! Original count: {len(my_chunks)}")
        except Exception as e:
            print(f"❌ Failed to read backup file: {e}")
            return
    else:
        print("⚠️ No backup detected, initiating fresh chunking and API summary logic...")
        my_chunks = ck.create_chunks_from_folder()
        if not my_chunks:
            print("❌ Failed to read TXT files.")
            return
        my_summaries = sm.generate_summaries(my_chunks)
        
        
        print("\n💾 Saving backup file just in case...")
        with open(backup_file, "wb") as f:
            pickle.dump({'chunks': my_chunks, 'summaries': my_summaries}, f)

    # Step 2: 
    print(f"\n🩺 Performing pre-ingestion data health check, removing invalid content...")
    
    summary_docs = []
    final_ids = []
    final_chunks = []
    
    for i, summary_text in enumerate(my_summaries):
        if summary_text and str(summary_text).strip() and summary_text != "[Error Summary]":
            source_file = my_chunks[i].metadata.get("source_document", "Unknown_Source")
            
            
            u_id = str(uuid.uuid4())
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
    print(f"✅ Done!")
    if skipped_count > 0:
        print(f"⚠️ Automatically filtered {skipped_count} empty summaries that would cause errors.")
    print(f"📦 Final valid ingestion count: {len(summary_docs)}")

    #Step 3:
    if not summary_docs:
        print("❌ No valid data available to store in the database.")
        return

    print("\n🚧 Writing to database in batches (ChromaDB & Docstore)...")
    batch_size = 500  
    total_len = len(summary_docs)
    
    

    for i in tqdm(range(0, total_len, batch_size), desc="Ingestion Progress"):
        end_idx = min(i + batch_size, total_len)
        
        curr_batch_docs = summary_docs[i:end_idx]
        curr_batch_ids = final_ids[i:end_idx]
        curr_batch_chunks = final_chunks[i:end_idx]
        
        try:
            # 1. Store in vector database
            emb.vectorstore.add_documents(curr_batch_docs)
            
            # 2. Store in original document store
            batch_bytes = [pickle.dumps(c) for c in curr_batch_chunks]
            emb.store.mset(list(zip(curr_batch_ids, batch_bytes)))
            
        except Exception as e:
            print(f"\n❌ Failed to write at batch {i} to {end_idx}: {e}")
            print("💡 Suggestion: If errors persist, try reducing the batch_size to 100.")
            return

    print("\n" + "="*40)
    print("🎉 Task completed successfully!")
    print(f"📁 Vector DB: ./chroma_db")
    print(f"📁 Document Store: ./docstore")
    print("="*40)

if __name__ == "__main__":
    build_database()
