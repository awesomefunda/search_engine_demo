"""Reranking: a feature scorer over the fused candidate set.

Retrieval is tuned for recall and is cheap per document. Reranking is allowed
to be expensive because it only ever sees ~50 candidates, so it can look at
things a bag-of-words retriever cannot: whether the query appears as an intact
phrase, whether it is in the title, how tightly the terms cluster, and how
much of the query the page actually covers.

Final score is a documented linear blend. A linear model is the right call for
an MVP: every contribution is inspectable, which matters more here than the
last few points of accuracy a learned ranker would buy.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import numpy as np

from .config import SETTINGS
from .fuse import Candidate, minmax
from .lexical import parse_query
from .semantic import EmbeddingIndex
from .store import PageRow, Store
from .textutil import min_window_span, raw_tokens, stem, tokenize

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Scored:
    page_id: int
    url: str
    title: str
    score: float
    snippet: str
    matched_terms: list[str]
    sources: list[str]
    features: dict[str, float] = field(default_factory=dict)
    near_duplicates: list[str] = field(default_factory=list)
    content_hash: str = ""      # internal: duplicate key, not part of the API

    def as_row(self) -> dict:
        return {
            "page_id": self.page_id,
            "url": self.url,
            "title": self.title,
            "score": round(self.score, 6),
            "snippet": self.snippet,
            "matched_terms": self.matched_terms,
            "sources": sorted(set(self.sources)),
            "features": {k: round(v, 4) for k, v in self.features.items()},
            "near_duplicates": self.near_duplicates,
        }


def collapse_duplicates(results: list[Scored]) -> list[Scored]:
    """Keep one result per identical extracted text.

    Some pages differ only in a query parameter the server ignores -- the
    industry pages switch tabs client side, so four URLs return byte-identical
    HTML. Without this, one page of results is four copies of one document.
    The best-scoring URL wins and the others are listed on it.
    """
    best: dict[str, Scored] = {}
    order: list[Scored] = []
    for r in results:
        if not r.content_hash:
            order.append(r)
            continue
        keeper = best.get(r.content_hash)
        if keeper is None:
            best[r.content_hash] = r
            order.append(r)
        else:
            keeper.near_duplicates.append(r.url)
    return order


def _phrase_feature(query: str, page: PageRow) -> float:
    """Graded exact-phrase signal. Title hit beats a body hit."""
    phrases, terms = parse_query(query)
    needles = [p.lower() for p in phrases]
    if len(terms) > 1:
        needles.append(" ".join(terms).lower())
    if not needles:
        return 0.0
    title = (page.title + " " + page.headings).lower()
    body = (page.meta_desc + " " + page.body).lower()
    best = 0.0
    for n in needles:
        if n in title:
            best = max(best, 1.0)
        elif n in body:
            best = max(best, 0.6)
    return best


def _overlap(query_stems: set[str], text: str) -> float:
    if not query_stems:
        return 0.0
    present = query_stems & set(tokenize(text))
    return len(present) / len(query_stems)


def _proximity_feature(query_stems: set[str], chunk: str) -> float:
    """1.0 when every query term sits inside a tight window, decaying with span."""
    if len(query_stems) < 2 or not chunk:
        return 0.0
    toks = [stem(t) for t in raw_tokens(chunk)]
    span = min_window_span(toks, query_stems)
    if span is None:
        return 0.0
    ideal = len(query_stems)
    return float(ideal / max(span, ideal))


def _authority_feature(page: PageRow, inlinks: dict[str, int], max_inlinks: int) -> float:
    """Shallow, well-linked pages are better default answers than deep leaves."""
    depth_part = 1.0 / (1.0 + max(0, page.depth))
    link_part = math.log1p(inlinks.get(page.url, 0)) / math.log1p(max_inlinks or 1)
    return 0.5 * depth_part + 0.5 * min(1.0, link_part)


def _make_snippet(query_stems: set[str], page: PageRow, chunks: list[str],
                  max_chars: int = 260) -> str:
    """Pick the passage with the most distinct query terms, trimmed to a sentence."""
    pool = chunks or []
    best_text, best_hits = "", -1
    for text in pool:
        hits = len(query_stems & set(tokenize(text)))
        if hits > best_hits:
            best_text, best_hits = text, hits
    if best_hits <= 0:
        best_text = page.meta_desc or page.body or page.title
    if not best_text:
        return ""

    sentences = _SENT_SPLIT.split(best_text)
    scored = [(len(query_stems & set(tokenize(s))), i, s) for i, s in enumerate(sentences)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    out: list[str] = []
    total = 0
    for _hits, i, s in sorted(scored[:3], key=lambda t: t[1]):
        s = s.strip()
        if not s:
            continue
        out.append(s)
        total += len(s)
        if total >= max_chars:
            break
    snippet = " ".join(out) or best_text
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "…"
    return snippet


class Reranker:
    def __init__(self, store: Store, embeddings: EmbeddingIndex):
        self.store = store
        self.embeddings = embeddings
        self._inlinks = store.inlink_counts()
        self._max_inlinks = max(self._inlinks.values(), default=1)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Scored]:
        if not candidates:
            return []

        pages = self.store.get_pages_by_ids([c.page_id for c in candidates])
        query_stems = set(tokenize(query))
        qvec = self.embeddings.embed_query(query)

        lex_norm = minmax([c.lexical_score for c in candidates])
        s = SETTINGS
        out: list[Scored] = []

        for c, lex in zip(candidates, lex_norm):
            page = pages.get(c.page_id)
            if page is None:
                continue
            chunks = self.store.chunks_for_page(page.id)

            sem_best = (self.embeddings.best_chunk_score(page.id, qvec)
                        if qvec.size else c.semantic_score)
            sem = float(np.clip(sem_best, 0.0, 1.0))
            phrase = _phrase_feature(query, page)
            title = _overlap(query_stems, page.title + " " + page.headings)
            coverage = _overlap(query_stems, " ".join(
                [page.title, page.headings, page.meta_desc, page.body]))
            best_chunk = ""
            if chunks:
                best_chunk = max(chunks, key=lambda t: len(query_stems & set(tokenize(t))))
            proximity = _proximity_feature(query_stems, best_chunk)
            authority = _authority_feature(page, self._inlinks, self._max_inlinks)

            # Only one retriever found it -> it has no cross-signal support.
            lex_eff = lex if c.lexical_rank is not None else 0.0

            score = (s.w_lexical * lex_eff + s.w_semantic * sem + s.w_phrase * phrase
                     + s.w_title * title + s.w_coverage * coverage
                     + s.w_proximity * proximity + s.w_authority * authority)

            out.append(Scored(
                page_id=page.id,
                url=page.url,
                title=page.title or page.url,
                score=score,
                snippet=_make_snippet(query_stems, page, chunks),
                matched_terms=sorted(query_stems & set(tokenize(
                    " ".join([page.title, page.headings, page.meta_desc, page.body])))),
                sources=c.sources,
                content_hash=page.content_hash,
                features={
                    "lexical": lex_eff, "semantic": sem, "phrase": phrase, "title": title,
                    "coverage": coverage, "proximity": proximity, "authority": authority,
                    "rrf": c.rrf,
                },
            ))

        out.sort(key=lambda r: (-r.score, r.page_id))
        return collapse_duplicates(out)
