"""HTML -> fielded document + outlinks + chunks."""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

from .config import SETTINGS
from .urls import in_scope, normalize

_WS = re.compile(r"\s+")
_DROP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe", "form")
_BOILERPLATE_TAGS = ("nav", "header", "footer", "aside")


@dataclass
class Document:
    url: str
    title: str = ""
    meta_desc: str = ""
    headings: str = ""
    body: str = ""
    outlinks: list[str] = field(default_factory=list)
    chunks: list[str] = field(default_factory=list)
    content_hash: str = ""


def _clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Overlapping word windows. Overlap keeps a phrase from being split in half."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(words), step):
        window = words[start:start + size]
        if not window:
            break
        out.append(" ".join(window))
        if start + size >= len(words):
            break
    return out


def parse(url: str, html: bytes | str) -> Document:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()

    doc = Document(url=url)
    doc.title = _clean(soup.title.get_text() if soup.title else "")

    for name in ("description", "og:description"):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if tag and tag.get("content"):
            doc.meta_desc = _clean(tag["content"])
            break

    # Links are collected BEFORE boilerplate is stripped: nav menus are the
    # cheapest way to discover a site's top-level pages.
    seen: set[str] = set()
    for a in soup.find_all("a", href=True):
        norm = normalize(a["href"], base=url)
        if norm and norm != url and in_scope(norm) and norm not in seen:
            seen.add(norm)
            doc.outlinks.append(norm)

    # Headings and body are taken AFTER it. A mega-menu is full of <h2>/<h3>,
    # and letting those into the document makes every page look alike and
    # fills snippets with navigation text.
    main = soup.find("main") or soup.find(attrs={"role": "main"}) or soup.body or soup
    for tag in main(list(_BOILERPLATE_TAGS)):
        tag.decompose()

    doc.headings = _clean(" | ".join(
        _clean(h.get_text()) for h in main.find_all(["h1", "h2", "h3"])[:40]
    ))
    doc.body = _clean(main.get_text(" "))

    # Hash the extracted text, not the raw HTML: a changed build id or a
    # rotating CSRF token in the markup should not count as a content change.
    payload = "\n".join([doc.title, doc.meta_desc, doc.headings, doc.body])
    doc.content_hash = hashlib.blake2b(payload.encode("utf-8", "ignore"),
                                       digest_size=16).hexdigest()

    # Chunks feed both semantic matching and snippet selection, so they are
    # built from prose only. The title is excluded: it is displayed separately
    # and already carries an 8x weight on the lexical side, and repeating it in
    # chunk 0 made it the snippet for most pages.
    chunk_source = " ".join(x for x in (doc.meta_desc, doc.body) if x)
    doc.chunks = chunk_words(chunk_source, SETTINGS.chunk_words, SETTINGS.chunk_overlap_words)
    return doc
