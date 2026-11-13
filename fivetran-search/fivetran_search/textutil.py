"""Tokenization shared by the semantic index and the reranker.

Kept deliberately small: lowercase word tokens, a short stopword list, and a
conservative suffix strip so "connectors" and "connector" collapse to one term.
FTS5 does its own Porter stemming on the lexical side; this only has to be
consistent with itself.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*", re.I)

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have
how i if in into is it its may me more most my no nor not of on or our out own
she so than that the their them then there these they this those to too us was
we were what when where which while who whom why will with would you your他
""".split())

_SUFFIXES = ("ization", "izations", "iveness", "ations", "ation", "ingly", "edly",
             "ments", "ment", "ness", "ions", "ing", "ers", "er", "ies", "ied",
             "es", "ed", "s")


def stem(token: str) -> str:
    """Crude, order-sensitive suffix strip. Never shortens below 4 characters."""
    for suf in _SUFFIXES:
        if len(token) - len(suf) >= 4 and token.endswith(suf):
            base = token[: -len(suf)]
            if suf == "ies":
                return base + "y"
            return base
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for m in _TOKEN_RE.finditer(text or ""):
        tok = m.group(0).lower().strip(".")
        if len(tok) < 2:
            continue
        if not keep_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out


def raw_tokens(text: str) -> list[str]:
    """Tokens with stopwords kept and no stemming: used for phrase matching."""
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")]


def min_window_span(tokens: list[str], terms: set[str]) -> int | None:
    """Shortest window of `tokens` containing every term in `terms`.

    Used as a proximity feature: query words appearing next to each other is
    a far stronger signal than the same words scattered across a long page.
    Returns None when not all terms are present.
    """
    if not terms:
        return None
    need = len(terms)
    have: dict[str, int] = {}
    best: int | None = None
    left = 0
    for right, tok in enumerate(tokens):
        if tok in terms:
            have[tok] = have.get(tok, 0) + 1
        while len(have) == need:
            span = right - left + 1
            if best is None or span < best:
                best = span
            lt = tokens[left]
            if lt in terms:
                have[lt] -= 1
                if have[lt] == 0:
                    del have[lt]
            left += 1
    return best
