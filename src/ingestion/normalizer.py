"""Convert raw scraped HTML into structured, section-aware documents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_sections(html: str) -> list[dict]:
    """Split HTML into sections by heading tags, preserving ancestry."""
    soup = BeautifulSoup(html, "html.parser")
    sections: list[dict] = []
    current_path: list[tuple[int, str]] = []
    current_content: list[str] = []

    def _save_section(path: list[tuple[int, str]], content: list[str]) -> None:
        if content:
            text = "\n".join(content).strip()
            if text:
                sections.append({
                    "section_path": [h for _, h in path],
                    "content": text,
                })

    heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}

    for element in soup.children:
        if not hasattr(element, "name") or element.name is None:
            continue
        if element.name in heading_tags:
            _save_section(current_path, current_content)
            current_content = []
            level = int(element.name[1])
            heading_text = element.get_text(strip=True)
            current_path = [(l, h) for l, h in current_path if l < level]
            current_path.append((level, heading_text))
        else:
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_content.append(text)

    _save_section(current_path, current_content)
    return sections


def normalize_doc(raw: dict) -> dict:
    """Convert a raw scraped doc to a structured normalized document."""
    content = _clean_text(raw.get("raw_html", ""))
    sections = _extract_sections(raw.get("raw_html", ""))

    meta = raw.get("metadata", {})

    return {
        "doc_id": raw["doc_id"],
        "title": raw.get("title", ""),
        "url": raw.get("url", ""),
        "version": meta.get("version"),
        "section_path": raw.get("section_path", []),
        "endpoint": meta.get("endpoint"),
        "method": meta.get("method"),
        "sdk": meta.get("sdk", []),
        "content": content,
        "sections": sections,
        "metadata": {
            "url_path": meta.get("url_path", ""),
            "path_parts": meta.get("path_parts", []),
        },
    }


def normalize_all(raw_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(raw_dir.glob("*.json")):
        raw = json.loads(path.read_text())
        docs.append(normalize_doc(raw))
    logger.info("Normalized %d documents", len(docs))
    return docs


def save_processed(docs: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        out_path = output_dir / f"{doc['doc_id']}.json"
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps([d["doc_id"] for d in docs], indent=2))
    logger.info("Saved %d processed docs to %s", len(docs), output_dir)
