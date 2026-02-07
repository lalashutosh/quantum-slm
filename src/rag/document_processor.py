import fitz  # PyMuPDF
import re
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles technical PDF extraction and LaTeX preprocessing."""

    LATEX_MAPPING = {
        r"\\langle": "inner product start ",
        r"\\rangle": " inner product end",
        r"\\phi": "phi",
        r"\\psi": "psi",
        r"\\kappa": "kernel function kappa",
        r"\\alpha": "weight coefficient alpha",
        r"\^2": " squared",
        r"K\(x, x'\)": "kernel function K of x and x prime",
        r"\|(.+?)\|": r"absolute value of \1",
    }

    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
        """Extracts text per page from PDF."""
        pages_content = []
        try:
            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                pages_content.append({
                    "text": page.get_text(),
                    "page": page_num + 1
                })
            return pages_content
        except Exception as e:
            logger.error(f"Failed to extract PDF {pdf_path}: {e}")
            return []

    def preprocess_latex(self, text: str) -> Tuple[str, List[str]]:
        """
        Converts LaTeX symbols to natural language for better embedding.
        Returns processed text and list of original equations.
        """
        original_equations = re.findall(r"\$.*?\$|\begin\{equation\}.*?\text\{equation\}", text, re.DOTALL)
        
        processed_text = text
        for pattern, replacement in self.LATEX_MAPPING.items():
            processed_text = re.sub(pattern, replacement, processed_text)
        
        return processed_text, original_equations

    @staticmethod
    def detect_sections(text: str) -> str:
        """Identifies which section the text likely belongs to."""
        sections = {
            "Abstract": r"(?i)abstract",
            "Introduction": r"(?i)introduction",
            "Methods": r"(?i)methods|methodology|quantum kernel",
            "Results": r"(?i)results|discussion",
            "Conclusion": r"(?i)conclusion"
        }
        for section, pattern in sections.items():
            if re.search(pattern, text[:500]):
                return section
        return "General"