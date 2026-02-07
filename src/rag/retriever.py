from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import List

class QuantumRetriever:
    """Handles vector storage and MMR retrieval logic."""

    def __init__(self, persist_path: str, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )
        self.persist_path = persist_path
        self.vectorstore = None

    def setup_vectorstore(self, documents: List):
        """Initializes or updates ChromaDB."""
        texts = [d["page_content"] for d in documents]
        metadatas = [d["metadata"] for d in documents]
        
        self.vectorstore = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            persist_directory=self.persist_path
        )

    def mmr_retrieve(self, query: str, k: int = 5, fetch_k: int = 20):
        """Executes Maximal Marginal Relevance search."""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory=self.persist_path, 
                embedding_function=self.embeddings
            )
            
        return self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": fetch_k,
                "lambda_mult": 0.7
            }
        ).invoke(query)