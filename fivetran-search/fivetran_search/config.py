"""Central configuration. Everything tunable lives here, nothing is hardcoded downstream."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

DATA_DIR = Path(os.environ.get("FTS_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

# --- URL pattern rules ------------------------------------------------------
# Deny wins over allow. These are the "patterns" that keep a 50-page budget
# spent on content pages instead of assets, locales and infinite facets.
DENY_PATH_PATTERNS = [
    r"\.(png|jpe?g|gif|svg|webp|ico|css|js|json|xml|pdf|zip|gz|mp4|webm|woff2?|ttf|eot)$",
    r"^/(ja|jp|de|fr|es|it|nl|pt|ko|zh)(/|$)",   # locale trees (robots also disallows /ja,/jp)
    r"^/(search|login|dashboard|cgi-bin|tmp|cache)(/|$)",
    r"^/blog/page/\d+",                            # blog pagination -> near-duplicate listings
    r"^/(cdn-cgi|_next|static|assets)(/|$)",
]
# Paths we prefer to spend budget on first (frontier priority, not a hard filter).
PRIORITY_PATH_PATTERNS = [
    r"^/$",
    r"^/(product|products|platform|solutions|pricing|connectors|integrations|docs|features)",
    r"^/(about|company|customers|security|why-fivetran)",
]
# Query params that never change the document (tracking noise) -> dropped on normalize.
DROP_QUERY_PARAM_PREFIXES = ("utm_", "gclid", "fbclid", "mc_", "ref", "hsa_", "_hs")


@dataclass(frozen=True)
class Settings:
    # crawl
    seed_urls: tuple[str, ...] = ("https://www.fivetran.com/",)
    allowed_hosts: tuple[str, ...] = ("fivetran.com", "www.fivetran.com")
    # apex and www are the same site, so they must collapse to one host for
    # dedup -- but which one is not cosmetic: fivetran.com returns 404 for most
    # paths that www.fivetran.com serves, so www is the canonical form here.
    canonical_host: str = "www.fivetran.com"
    max_pages: int = 50
    max_depth: int = 4
    request_timeout: float = 15.0
    crawl_delay: float = 0.5          # politeness floor between requests, seconds
    user_agent: str = "mini-search-bot/0.1 (+internal search MVP; respects robots.txt)"
    max_bytes_per_page: int = 2_000_000
    fetch_attempts: int = 2           # one retry on a transient TLS/connection error
    retry_backoff: float = 1.0
    # Bump when extract.py changes what it pulls out of a page. Stored documents
    # are then stale even though the site is not, so the next crawl must refetch
    # without validators instead of accepting 304s against the old parse.
    parser_version: int = 2

    # freshness / caching
    page_ttl_seconds: int = 24 * 3600          # a page older than this is refetched (conditionally)
    corpus_ttl_seconds: int = 24 * 3600        # within TTL + enough pages => skip crawling entirely
    query_cache_ttl_seconds: int = 15 * 60     # ranked-snapshot lifetime
    min_corpus_pages: int = 20                 # below this the corpus counts as incomplete

    # chunking
    chunk_words: int = 110
    chunk_overlap_words: int = 30

    # retrieval
    lexical_candidates: int = 60
    semantic_candidates: int = 60
    fusion_candidates: int = 50     # how many survive fusion into the reranker
    rrf_k: int = 60                 # RRF damping constant
    lsa_components: int = 96
    max_vocab: int = 8000
    min_df: int = 2

    # rerank weights (documented in README)
    w_lexical: float = 0.30
    w_semantic: float = 0.26
    w_phrase: float = 0.16
    w_title: float = 0.12
    w_coverage: float = 0.10
    w_proximity: float = 0.04
    w_authority: float = 0.02

    # serving
    page_size: int = 10
    db_path: Path = field(default_factory=lambda: DATA_DIR / "index.db")
    cache_dir: Path = field(default_factory=lambda: DATA_DIR / "cache")


SETTINGS = Settings()

DENY_RE = [re.compile(p, re.I) for p in DENY_PATH_PATTERNS]
PRIORITY_RE = [re.compile(p, re.I) for p in PRIORITY_PATH_PATTERNS]


def path_is_denied(path: str) -> bool:
    return any(r.search(path) for r in DENY_RE)


def path_priority(path: str) -> int:
    """Lower sorts first in the frontier."""
    for i, r in enumerate(PRIORITY_RE):
        if r.search(path):
            return i
    return len(PRIORITY_RE) + path.count("/")
