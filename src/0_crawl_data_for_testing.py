import os
import time
from urllib.parse import urlencode
import re
import requests
from bs4 import BeautifulSoup

root = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
BASE_URL = "https://bioenergykdf.ornl.gov/biokdf-search"
ENTRY_PREFIX = "https://bioenergykdf.ornl.gov"
OUTPUT_DIR = os.path.join(root, "data science kdf", "data")
REQUEST_DELAY = 1.5  # seconds between requests

def make_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_search_page(page_num=0):
    """Fetch a search result page."""
    params = {"page": page_num}
    r = requests.get(BASE_URL, params=params)
    r.raise_for_status()
    return r.text

def parse_search_page(html):
    """Extract all /document/ links and titles from one page."""
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for a in soup.select("a[href^='/document/']"):
        title = a.get_text(strip=True)
        link = a["href"]
        if link.startswith("/"):
            link = ENTRY_PREFIX + link
        results.append((title, link))
    print(f"  Found {len(results)} document links.")
    return results

def fetch_entry_html(url):
    """Download and save the document page."""
    r = requests.get(url)
    r.raise_for_status()
    fname = url.split("/")[-1] + ".html"
    path = os.path.join(OUTPUT_DIR, fname)
    with open(path, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"Saved: {path}")


def crawl_all(keyword="",page_start=0, max_pages=None, max_docs=None):
    """
    Crawl all search result pages and download up to `max_docs` document pages.
    - page_start: starting search results page index (default 0)
    - max_pages: max number of search pages to iterate (None = until empty)
    - max_docs: max number of document pages to save (None = no limit)
    """
    total_docs = 0
    page_num = page_start

    while True:
        print(f"\nFetching search page {page_num} ...")
        html = get_search_page(page_num)
        items = parse_search_page(html)
        if not items:
            print("No more items found, stopping.")
            break

        for title, link in items:
            print(f"  Crawling: {title}")
            fetch_entry_html(link)
            total_docs += 1
            time.sleep(1.5)  # polite delay

            # stop when user-specified doc limit reached
            if max_docs and total_docs >= max_docs:
                print(f"\nReached max_docs={max_docs}, stopping crawl.")
                return

        page_num += 1
        if max_pages and page_num >= max_pages:
            print(f"\nReached max_pages={max_pages}, stopping crawl.")
            break

if __name__ == "__main__":

    crawl_all(page_start=0, max_pages=None, max_docs=10)