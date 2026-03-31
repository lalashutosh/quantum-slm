import arxiv
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def fetch_top_qml_papers(queries, per_query_top=3, total_desired=50, min_year=2020, fetch_per_query=50):
    """
    Fetch top Quantum Machine Learning papers from arXiv.

    Args:
        queries (list[str]): list of search queries
        per_query_top (int): how many top papers per query to pick
        total_desired (int): total number of papers to return
        min_year (int): minimum publication year to include
        fetch_per_query (int): how many papers to fetch per query before filtering
    """
    all_candidates = []

    logging.info(f"Fetching up to {fetch_per_query} papers per query...")
    
    for q_idx, q in enumerate(queries):
        logging.info(f"Query {q_idx+1}/{len(queries)}: {q}")
        search = arxiv.Search(
            query=q,
            max_results=fetch_per_query,
            sort_by=arxiv.SortCriterion.Relevance
        )
        try:
            for result in search.results():
                if result.published.year < min_year:
                    continue
                # Keep track of which query this came from
                all_candidates.append({
                    "title": result.title,
                    "pdf_url": result.pdf_url,
                    "summary": result.summary,
                    "year": result.published.year,
                    "query_idx": q_idx
                })
        except Exception as e:
            logging.error(f"Error fetching query '{q}': {e}")

    logging.info(f"Collected {len(all_candidates)} papers after filtering by year >= {min_year}.")

    # Step 1: pick top N per query
    selected_papers = []
    for q_idx in range(len(queries)):
        papers_for_query = [p for p in all_candidates if p["query_idx"] == q_idx]
        selected_papers.extend(papers_for_query[:per_query_top])

    logging.info(f"Selected {len(selected_papers)} papers from per-query top {per_query_top}.")

    # Step 2: fill remaining papers by relevance + recency
    remaining_needed = total_desired - len(selected_papers)
    if remaining_needed > 0:
        remaining_candidates = [p for p in all_candidates if p not in selected_papers]
        # Sort by year descending (more recent first)
        remaining_candidates.sort(key=lambda x: x["year"], reverse=True)
        selected_papers.extend(remaining_candidates[:remaining_needed])

    # Deduplicate by title
    unique_titles = set()
    final_papers = []
    for p in selected_papers:
        if p["title"] not in unique_titles:
            final_papers.append(p)
            unique_titles.add(p["title"])

    logging.info(f"Final paper count after deduplication: {len(final_papers)}")
    return final_papers



import os
import requests
from urllib.parse import urlsplit

# Directory to save PDFs
SAVE_DIR = "papers"
os.makedirs(SAVE_DIR, exist_ok=True)

def sanitize_filename(title):
    """Replace illegal filename characters."""
    keepchars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keepchars else "_" for c in title)

def download_papers(papers):
    for idx, paper in enumerate(papers, 1):
        title = paper["title"]
        pdf_url = paper["pdf_url"]
        filename = sanitize_filename(title) + ".pdf"
        filepath = os.path.join(SAVE_DIR, filename)

        # Skip if already downloaded
        if os.path.exists(filepath):
            print(f"[{idx}] Skipping already downloaded: {title}")
            continue

        print(f"[{idx}] Downloading: {title}")
        try:
            r = requests.get(pdf_url, stream=True, timeout=30)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print(f"   Saved to {filepath}")
        except Exception as e:
            print(f"   Error downloading {title}: {e}")




if __name__ == "__main__":
    queries = [
        '(ti:"quantum neural network" OR abs:"quantum neural network")',
        '(ti:"quantum kernel" OR abs:"quantum kernel")',
        '(ti:"quantum reinforcement learning" OR abs:"quantum reinforcement learning")',
        '(ti:"Variational Quantum Eigensolver" OR abs:"Variational Quantum Eigensolver")',
        '(ti:"Quantum Support Vector Machines" OR abs:"Quantum Support Vector Machines")',
        '(ti:"quantum kernels" OR abs:"quantum kernels")',
        '(ti:"quantum machine learning" OR abs:"quantum machine learning")',
        '(ti:"Parameterized Quantum Circuits" OR abs:"Parameterized Quantum Circuits")',
        '(ti:"Quantum Feature Maps" OR abs:"Quantum Feature Maps")',
        '(ti:"NISQ" OR abs:"NISQ")'
    ]

    papers = fetch_top_qml_papers(queries, per_query_top=3, total_desired=50)

    print("\n=== Top 50 QML Papers ===")
    for i, p in enumerate(papers, 1):
        print(f"[{i}] {p['title']} ({p['year']})\n    PDF: {p['pdf_url']}")


    download_papers(papers)