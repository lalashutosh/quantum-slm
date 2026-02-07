import numpy as np
from typing import List, Dict

class RetrievalEvaluator:
    """Measures retrieval performance (Recall, Precision, MRR)."""

    @staticmethod
    def evaluate(test_cases: List[Dict], retriever_results: List[List]) -> Dict:
        recalls = []
        mrrs = []
        
        for test, results in zip(test_cases, retriever_results):
            expected_id = test["expected_paper"]
            retrieved_ids = [res.metadata.get("paper_id") for res in results]
            
            # Recall@K
            recalls.append(1 if expected_id in retrieved_ids else 0)
            
            # MRR
            try:
                rank = retrieved_ids.index(expected_id) + 1
                mrrs.append(1 / rank)
            except ValueError:
                mrrs.append(0)
                
        return {
            "recall@5": np.mean(recalls),
            "mrr": np.mean(mrrs)
        }