import sys
import os
import glob
import torch

# PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from src.rag.rag_system import QuantumRAG

# 1. CONFIGURATION
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "models/v1_2_weights"
DB_PATH = "./rag_db"
PAPERS_DIR = "data/raw/pdfs" 

def main():
    print("Initializing Quantum-RAG System...")
    rag = QuantumRAG(persist_path=DB_PATH, embedding_model="BAAI/bge-large-en-v1.5")

    # --- PHASE 1: DYNAMIC INDEXING ---
    # We check if the database exists. If not, we walk the directory.
    if not os.path.exists(DB_PATH):
        # Find all PDF files in the directory
        pdf_files = glob.glob(os.path.join(PAPERS_DIR, "*.pdf"))
        
        if not pdf_files:
            print(f"Error: No PDF files found in {PAPERS_DIR}. Please check the folder.")
            return

        print(f"Found {len(pdf_files)} papers. Starting dynamic ingestion...")
        
        for pdf_path in pdf_files:
            # Generate paper_id from filename (e.g., 'havlicek_2019.pdf' -> 'havlicek_2019')
            paper_id = os.path.splitext(os.path.basename(pdf_path))[0]
            print(f"Processing: {paper_id}")
            
            rag.index_paper(
                pdf_path=pdf_path, 
                paper_id=paper_id, 
                metadata={"source": "research_corpus"}
            )
        print("Vector database created successfully.")
    else:
        print("Vector database found. Skipping indexing.")

    # --- PHASE 2: LOAD FINE-TUNED SLM (4-bit) ---
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

    # --- PHASE 3: RESEARCH QUERY ---
    query = "Based on the experiments, what happens to the SVM accuracy when coherent noise reaches the pi/6 threshold?"
    
    print(f"\n Querying RAG System: {query}")
    response = rag.generate_with_context(model, tokenizer, query)
    
    print("\n" + "="*50)
    print(" RAG-AUGMENTED RESEARCH ASSISTANT:")
    print("="*50)
    print(response)
    print("="*50)

if __name__ == "__main__":
    main()
