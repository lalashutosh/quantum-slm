from .document_processor import DocumentProcessor
from .chunking import QuantumChunker
from .retriever import QuantumRetriever
from .evaluator import RetrievalEvaluator

class QuantumRAG:
    """Main Orchestrator for Quantum Research RAG."""

    def __init__(self, persist_path: str, embedding_model: str):
        self.processor = DocumentProcessor()
        self.chunker = QuantumChunker()
        self.retriever = QuantumRetriever(persist_path, embedding_model)

    def index_paper(self, pdf_path: str, paper_id: str, metadata: dict):
        """Pipeline to process and index a new paper."""
        raw_pages = self.processor.extract_text_from_pdf(pdf_path)
        all_chunks = []
        
        for page in raw_pages:
            proc_text, latex_list = self.processor.preprocess_latex(page["text"])
            
            page_meta = metadata.copy()
            page_meta.update({
                "paper_id": paper_id,
                "page": page["page"],
                "section": self.processor.detect_sections(page["text"]),
                "original_latex": str(latex_list[:5]) # Store sample for metadata
            })
            
            chunks = self.chunker.chunk_document(proc_text, page_meta)
            all_chunks.extend(chunks)
            
        self.retriever.setup_vectorstore(all_chunks)
        print(f"✅ Indexed {len(all_chunks)} chunks for {paper_id}")

    def generate_with_context(self, model, tokenizer, query: str):
        """Retrieves context and generates a stable, non-repeating response."""
        context_docs = self.retriever.mmr_retrieve(query)
        context_text = "\n\n".join([d.page_content for d in context_docs])
        
        # Refined Prompt to prevent looping
        template = f"""You are a precise technical research assistant. 
        Use the context below to answer the question. 
        If the information is not in the context, say so. 
        Do not repeat yourself.

        Context:
        {context_text}

        Question: {query}

        Answer:"""

        inputs = tokenizer(template, return_tensors="pt").to("cuda")
        
        # Add repetition_penalty and top_p for stability
        outputs = model.generate(
            **inputs, 
            max_new_tokens=250, 
            temperature=0.1, 
            repetition_penalty=1.1, # <--- Stops the looping
            top_p=0.9,               # <--- Adds a bit of diversity
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_text.split("Answer:")[-1].strip()