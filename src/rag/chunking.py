from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict

class QuantumChunker:
    """Splits documents into semantic chunks while preserving math integrity."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_document(self, text: str, doc_metadata: Dict) -> List[Dict]:
        """Chunks text and attaches metadata."""
        chunks = self.splitter.split_text(text)
        processed_chunks = []
        
        for chunk in chunks:
            # Metadata requirement check
            meta = doc_metadata.copy()
            meta["has_equation"] = "$" in chunk or r"\begin" in chunk
            
            processed_chunks.append({
                "page_content": chunk,
                "metadata": meta
            })
            
        return self.ensure_equation_integrity(processed_chunks)

    def ensure_equation_integrity(self, chunks: List[Dict]) -> List[Dict]:
        """Simple check to ensure math tags are closed within a chunk."""
        for chunk in chunks:
            content = chunk["page_content"]
            if content.count("$") % 2 != 0:
                # If an equation is cut off, we flag it for high-level retrieval logic
                chunk["metadata"]["math_fragment"] = True
        return chunks