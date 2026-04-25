"""Fetch Stripe docs pages and pull out headings, endpoints, and version metadata."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

STRIPE_DOCS_URLS = [
    "https://stripe.com/docs/api",
    "https://stripe.com/docs/api/charges",
    "https://stripe.com/docs/api/customers",
    "https://stripe.com/docs/api/payment_intents",
    "https://stripe.com/docs/api/payment_methods",
    "https://stripe.com/docs/api/refunds",
    "https://stripe.com/docs/api/subscriptions",
    "https://stripe.com/docs/api/invoices",
    "https://stripe.com/docs/api/products",
    "https://stripe.com/docs/api/prices",
    "https://stripe.com/docs/api/events",
    "https://stripe.com/docs/api/webhooks",
    "https://stripe.com/docs/api/errors",
    "https://stripe.com/docs/api/authentication",
    "https://stripe.com/docs/api/pagination",
    "https://stripe.com/docs/api/expanding_objects",
    "https://stripe.com/docs/api/idempotent_requests",
    "https://stripe.com/docs/api/metadata",
    "https://stripe.com/docs/api/versioning",
    "https://stripe.com/docs/keys",
    "https://stripe.com/docs/webhooks",
    "https://stripe.com/docs/payments/payment-intents",
    "https://stripe.com/docs/payments/accept-a-payment",
    "https://stripe.com/docs/billing/subscriptions/overview",
    "https://stripe.com/docs/radar/rules",
]

_HTTP_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b")
_ENDPOINT_RE = re.compile(r"/v\d[/\w{}]+")
_VERSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _make_doc_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def _extract_heading_hierarchy(soup: BeautifulSoup) -> list[str]:
    """Walk h1-h6 elements in order to build the current section ancestry."""
    path: list[str] = []
    current_levels: dict[int, str] = {}
    for tag in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(tag.name[1])
        text = tag.get_text(strip=True)
        current_levels[level] = text
        for k in list(current_levels.keys()):
            if k > level:
                del current_levels[k]
        path = [current_levels[l] for l in sorted(current_levels)]
    return path


def _extract_metadata(soup: BeautifulSoup, url: str) -> dict:
    """Extract API version, endpoint, HTTP method, SDK labels from page."""
    text = soup.get_text(" ")

    version_match = _VERSION_RE.search(text)
    version = version_match.group(0) if version_match else None

    method_match = _HTTP_METHOD_RE.search(text)
    method = method_match.group(0) if method_match else None

    endpoint_match = _ENDPOINT_RE.search(text)
    endpoint = endpoint_match.group(0) if endpoint_match else None

    sdk_labels = []
    for lang in ("curl", "python", "ruby", "php", "java", "node", "go", ".net"):
        if lang.lower() in text.lower():
            sdk_labels.append(lang)

    path_parts = urlparse(url).path.strip("/").split("/")

    return {
        "version": version,
        "method": method,
        "endpoint": endpoint,
        "sdk": sdk_labels,
        "url_path": urlparse(url).path,
        "path_parts": path_parts,
    }


def scrape_page(url: str, client: httpx.Client) -> dict | None:
    try:
        resp = client.get(url, timeout=20, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1") or soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", class_=re.compile(r"content|main|docs", re.I))
        or soup.body
    )
    if main is None:
        return None

    section_path = _extract_heading_hierarchy(soup)
    metadata = _extract_metadata(soup, url)

    raw_html = str(main)

    return {
        "doc_id": _make_doc_id(url),
        "url": url,
        "title": title,
        "section_path": section_path,
        "raw_html": raw_html,
        "metadata": metadata,
    }


def scrape_all(urls: list[str] | None = None, delay: float = 0.5) -> list[dict]:
    target_urls = urls or STRIPE_DOCS_URLS
    results: list[dict] = []

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AdaptiveCRAGBot/1.0; +https://github.com/research)",
        "Accept": "text/html,application/xhtml+xml",
    }

    with httpx.Client(headers=headers) as client:
        for url in target_urls:
            logger.info("Scraping %s", url)
            doc = scrape_page(url, client)
            if doc:
                results.append(doc)
            time.sleep(delay)

    logger.info("Scraped %d pages", len(results))
    return results


def save_raw(docs: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        out_path = output_dir / f"{doc['doc_id']}.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    logger.info("Saved %d raw docs to %s", len(docs), output_dir)
