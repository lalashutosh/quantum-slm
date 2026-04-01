import sys
import os
import glob
import torch

# PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.rag.agentic_rag import AgenticQuantumRAG

# 1. CONFIGURATION (Matching V1 setup)
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "models/v1_2_weights"
DB_PATH = "./rag_db"
PAPERS_DIR = "data/raw/pdfs" 

def main():
    print("🚀 Initializing Agentic Quantum-RAG System...")
    
    # --- PHASE 1: LOAD FINE-TUNED SLM (4-bit) ---
    print("Loading Specialized SLM (4-bit QLoRA)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

    # --- PHASE 2: INITIALIZE AGENTIC RAG ---
    # AgenticQuantumRAG needs the model and tokenizer at init to build the LangChain pipeline
    rag = AgenticQuantumRAG(
        persist_path=DB_PATH, 
        embedding_model="BAAI/bge-large-en-v1.5",
        model=model,
        tokenizer=tokenizer
    )

    # --- PHASE 3: DYNAMIC INDEXING (Check if DB exists) ---
    if not os.path.exists(DB_PATH):
        pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF files found in {PAPERS_DIR}. Please check the folder.")
            return

        print(f"Found {len(pdf_files)} papers. Starting dynamic ingestion...")
        for pdf_path in pdf_files:
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"Processing: {paper_id}")
            rag.index_paper(
                pdf_path=pdf_path, 
                paper_id=paper_id, 
                metadata={"source": "research_corpus"}
            )
        print("Vector database created successfully.")
    else:
        print("Vector database found. Ready for Agentic Reasoning.")

    # --- PHASE 4: AGENTIC RESEARCH QUERY ---
    query = "Search the database for experiments on SVM accuracy and explain what happens when coherent noise reaches the pi/6 threshold."
    
    print(f"\n🧠 Querying Agent: {query}")
    response = rag.generate_agentic_response(query)
    
    print("\n" + "="*50)
    print(" 🤖 AGENTIC RESEARCH ASSISTANT RESPONSE:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main()
