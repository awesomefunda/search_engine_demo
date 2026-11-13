"""Keyword retrieval: user text -> FTS5 MATCH expression -> BM25 hits.

Recall-first: terms are OR-ed so a long sentence still finds pages that match
only part of it. Precision is restored later by the reranker, which is the
usual retrieve-wide / rank-narrow split.
"""
from __future__ import annotations

import re

from .store import Store
from .textutil import STOPWORDS, raw_tokens

_QUOTED = re.compile(r'"([^"]+)"')
_FTS_SAFE = re.compile(r"[a-z0-9]+", re.I)


def _fts_term(token: str) -> str:
    # FTS5 treats most punctuation as syntax; quoting makes a token a literal.
    cleaned = " ".join(_FTS_SAFE.findall(token))
    return f'"{cleaned}"' if cleaned else ""


def parse_query(raw: str) -> tuple[list[str], list[str]]:
    """Returns (phrases, terms).

    Anything the user wrapped in double quotes becomes a required phrase;
    everything else becomes individual terms.
    """
    phrases = [p.strip() for p in _QUOTED.findall(raw) if p.strip()]
    rest = _QUOTED.sub(" ", raw)
    tokens = [t for t in raw_tokens(rest) if len(t) > 1]
    content = [t for t in tokens if t not in STOPWORDS]
    # A query made entirely of stopwords ("how do i do it") still needs terms.
    terms = content or tokens
    return phrases, terms


def build_fts_expression(raw: str) -> str:
    phrases, terms = parse_query(raw)
    clauses: list[str] = []

    for p in phrases:
        t = _fts_term(p)
        if t:
            clauses.append(t)
    required = " AND ".join(clauses) if clauses else ""

    or_terms = [t for t in (_fts_term(x) for x in terms) if t]
    optional = ""
    if or_terms:
        # The whole input as a phrase is also offered to FTS: documents
        # containing it intact score far better under BM25.
        if len(or_terms) > 1:
            whole = _fts_term(" ".join(terms))
            if whole:
                or_terms = [whole] + or_terms
        optional = "(" + " OR ".join(or_terms) + ")"

    if required and optional:
        return f"{required} AND {optional}"
    return required or optional


class LexicalRetriever:
    def __init__(self, store: Store):
        self.store = store

    def search(self, raw_query: str, limit: int) -> list[tuple[int, float]]:
        expr = build_fts_expression(raw_query)
        if not expr:
            return []
        hits = self.store.bm25_search(expr, limit)
        if hits:
            return hits
        # Fallback: drop the required-phrase constraint and retry OR-only.
        _phrases, terms = parse_query(raw_query)
        or_terms = [t for t in (_fts_term(x) for x in terms) if t]
        if not or_terms:
            return []
        return self.store.bm25_search("(" + " OR ".join(or_terms) + ")", limit)
