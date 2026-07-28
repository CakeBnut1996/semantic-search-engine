import os
from pathlib import Path
import re
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from typing import List, Any, Tuple


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url.strip())
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return url.strip()
    return ""


def _extract_original_url(html: str, soup: BeautifulSoup) -> str:
    # Browser-saved pages can include this marker comment.
    saved_from_match = re.search(r"saved from url=\(\d+\)(https?://[^\s\"'>]+)", html, re.IGNORECASE)
    if saved_from_match:
        candidate = _normalize_url(saved_from_match.group(1))
        if candidate:
            return candidate

    canonical = soup.find("link", rel=lambda v: v and "canonical" in " ".join(v).lower() if isinstance(v, list) else "canonical" in str(v).lower())
    if canonical and canonical.get("href"):
        candidate = _normalize_url(canonical.get("href", ""))
        if candidate:
            return candidate

    for attrs in (
        {"property": "og:url"},
        {"name": "og:url"},
        {"property": "twitter:url"},
        {"name": "twitter:url"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            candidate = _normalize_url(tag.get("content", ""))
            if candidate:
                return candidate

    return "Unknown Source"


def extract_text_and_url_from_html(path: str) -> Tuple[str, str, str]:
    if not os.path.exists(path):
        return "", "Unknown Source", "Untitled"

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()

    # 1. Parse and extract URL/title metadata.
    soup = BeautifulSoup(html, "html.parser")
    original_url = _extract_original_url(html, soup)
    source_title = soup.title.get_text(strip=True) if soup.title else Path(path).stem

    # 2. Extract visible text.
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()

    text = soup.get_text(separator="\n", strip=True)
    return text, original_url, source_title

# --- Text Processing Functions ---

def extract_text_from_html(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.extract()
    return soup.get_text(separator="\n", strip=True)


def clean_text(text: str) -> str:
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\f', '', text)
    return text.strip()


def filter_noise(text: str) -> str:
    lines = text.split("\n")
    clean_lines = []
    for ln in lines:
        s = ln.strip()
        if not s: continue
        if re.match(r'^\d+[\.\)]', s): continue
        if len(s) < 30 and s.isupper(): continue
        if "REFERENCES" in s.upper() or "TABLE" in s.upper(): continue
        clean_lines.append(ln)
    return "\n".join(clean_lines)