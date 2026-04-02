import arxiv
import logging
import os
import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SAVE_DIR = "papers"
META_FILE = "metadata.json"

os.makedirs(SAVE_DIR, exist_ok=True)

# Thread-safe structures
seen_titles = set()
metadata = []
lock = Lock()


def sanitize_filename(title):
    keepchars = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(c if c in keepchars else "_" for c in title)


def is_relevant_qml(paper):
    text = (paper.title + " " + paper.summary).lower()

    if "quantum" not in text:
        return False

    ml_terms = [
        "machine learning", "neural", "kernel",
        "learning", "reinforcement", "classification",
        "feature map"
    ]

    return any(term in text for term in ml_terms)


def load_existing_metadata():
    global metadata, seen_titles
    if os.path.exists(META_FILE):
        with open(META_FILE, "r") as f:
            metadata = json.load(f)
            seen_titles = set(m["title"] for m in metadata)
        logging.info(f"Loaded {len(metadata)} existing metadata entries.")


def save_metadata():
    with lock:
        with open(META_FILE, "w") as f:
            json.dump(metadata, f, indent=2)


def download_pdf(paper, retries=3):
    title = paper.title
    pdf_url = paper.pdf_url
    year = paper.published.year

    filename = sanitize_filename(title) + ".pdf"
    filepath = os.path.join(SAVE_DIR, filename)

    if os.path.exists(filepath):
        return None

    for attempt in range(retries):
        try:
            r = requests.get(
                pdf_url,
                stream=True,
                timeout=(5, 20)  # (connect timeout, read timeout)
            )
            r.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"Downloaded: {title}")
            return {
                "title": title,
                "pdf_url": pdf_url,
                "year": year,
                "file": filepath
            }

        except Exception as e:
            logging.warning(f"Retry {attempt+1} failed: {title}")

            time.sleep(2)

    logging.error(f"Failed permanently: {title}")
    return None


def fetch_and_download(target_size=2000, min_year=2020, max_workers=8):
    queries = [
        '(ti:"quantum machine learning" OR abs:"quantum machine learning")',
        '(ti:"quantum neural network" OR abs:"quantum neural network")',
        '(ti:"quantum kernel" OR abs:"quantum kernel")',
        '(ti:"quantum reinforcement learning" OR abs:"quantum reinforcement learning")',
        '(ti:"parameterized quantum circuit" OR abs:"parameterized quantum circuit")',
        '(ti:"variational quantum" OR abs:"variational quantum")',
        '(ti:"quantum feature map" OR abs:"quantum feature map")',
        '(ti:"quantum classification" OR abs:"quantum classification")',
        '(ti:"quantum deep learning" OR abs:"quantum deep learning")',
        '(ti:"NISQ machine learning" OR abs:"NISQ machine learning")'
    ]

    client = arxiv.Client()
    futures = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        for q_idx, query in enumerate(queries):
            logging.info(f"\nQuery {q_idx+1}/{len(queries)}: {query}")

            search = arxiv.Search(
                query=query,
                max_results=500,
                sort_by=arxiv.SortCriterion.Relevance
            )

            try:
                for result in client.results(search):

                    if result.published.year < min_year:
                        continue

                    if not is_relevant_qml(result):
                        continue

                    with lock:
                        if result.title in seen_titles:
                            continue
                        seen_titles.add(result.title)

                    # Submit download task
                    futures.append(executor.submit(download_pdf, result))

                    # Stop submitting if we hit target
                    if len(seen_titles) >= target_size:
                        break

                time.sleep(0.5)  # rate limit safety

            except Exception as e:
                logging.error(f"Error in query '{query}': {e}")

        # Collect results as they complete
        total_tasks = len(futures)
        completed = 0

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result:
                with lock:
                    metadata.append(result)

            if completed % 10 == 0:
                logging.info(f"Progress: {completed}/{total_tasks} completed")

            if completed % 20 == 0:
                save_metadata()

    save_metadata()
    logging.info(f"\nDone. Total downloaded: {len(metadata)}")


if __name__ == "__main__":
    load_existing_metadata()
    fetch_and_download(target_size=2000, max_workers=4)