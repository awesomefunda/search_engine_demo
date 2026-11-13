"""SearchService: the facade the UI, the API and the CLI all talk to.

One method does the whole thing, and every expensive stage is behind a cache:

    search(query, cursor)
      -> ensure_corpus()        crawl only if the corpus is missing or stale
      -> ensure_index()         rebuild LSA only if corpus_version changed
      -> query snapshot?        reuse the ranked list if this query is cached
           miss: retrieve (BM25 + LSA) -> RRF fuse -> rerank -> snapshot
      -> keyset page from the snapshot

A page-2 request therefore costs one SQLite read: no HTTP, no SVD, no scoring.
"""
from __future__ import annotations

import hashlib
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import SETTINGS, Settings
from .crawler import CrawlReport, Crawler
from .fetcher import Fetcher
from .fuse import reciprocal_rank_fusion
from .lexical import LexicalRetriever
from .pagination import Cursor, InvalidCursor, page_after, sort_key
from .rerank import Reranker
from .semantic import EmbeddingIndex
from .store import Store

_WS = re.compile(r"\s+")


def normalize_query(raw: str) -> str:
    return _WS.sub(" ", (raw or "").strip()).lower()


def query_key(normalized: str) -> str:
    return hashlib.blake2b(normalized.encode(), digest_size=12).hexdigest()


@dataclass
class SearchResponse:
    query: str
    normalized_query: str
    results: list[dict] = field(default_factory=list)
    total_results: int = 0
    page_size: int = SETTINGS.page_size
    offset: int = 0
    next_cursor: str | None = None
    prev_cursor: str | None = None
    has_prev: bool = False
    cursor_stale: bool = False
    corpus_version: str = ""
    cache: dict = field(default_factory=dict)
    timings_ms: dict = field(default_factory=dict)
    corpus: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


class SearchService:
    def __init__(self, settings: Settings = SETTINGS, store: Store | None = None,
                 crawler: Crawler | None = None):
        self.settings = settings
        self.store = store or Store(Path(settings.db_path))
        self.crawler = crawler or Crawler(self.store, Fetcher())
        self.lexical = LexicalRetriever(self.store)
        self._embeddings: EmbeddingIndex | None = None
        self._index_lock = threading.RLock()
        self._search_lock = threading.RLock()

    # ---------------- corpus + index lifecycle -----------------------------
    def ensure_corpus(self, force: bool = False) -> CrawlReport:
        return self.crawler.ensure_corpus(force=force)

    def ensure_index(self, corpus_version: str) -> tuple[EmbeddingIndex, str]:
        """Returns (index, cache_state) where cache_state is memory|disk|rebuilt."""
        with self._index_lock:
            if self._embeddings is not None and self._embeddings.corpus_version == corpus_version:
                return self._embeddings, "memory"

            cache_dir = Path(self.settings.cache_dir)
            loaded = EmbeddingIndex.load(cache_dir, corpus_version)
            if loaded is not None:
                self._embeddings = loaded
                return loaded, "disk"

            index = EmbeddingIndex().build(self.store.iter_chunks(), corpus_version)
            if index.n_chunks:
                index.save(cache_dir)
            self._embeddings = index
            return index, "rebuilt"

    # ---------------- the query path ---------------------------------------
    def search(self, raw_query: str, cursor_token: str | None = None,
               page_size: int | None = None, force_crawl: bool = False) -> SearchResponse:
        t0 = time.perf_counter()
        page_size = page_size or self.settings.page_size
        normalized = normalize_query(raw_query)
        resp = SearchResponse(query=raw_query, normalized_query=normalized, page_size=page_size)

        if not normalized:
            resp.warnings.append("empty query")
            resp.timings_ms = {"total": round((time.perf_counter() - t0) * 1000, 1)}
            return resp

        qkey = query_key(normalized)

        with self._search_lock:
            # A paginating request must never trigger a crawl: it has to see the
            # same corpus the first page was ranked against.
            t = time.perf_counter()
            if cursor_token:
                crawl_report = CrawlReport(
                    pages_in_corpus=self.store.page_count(),
                    corpus_version=self.store.compute_corpus_version(),
                    skipped_entirely=True,
                )
                crawl_report.notes.append("crawl skipped: paginating an existing snapshot")
            else:
                crawl_report = self.ensure_corpus(force=force_crawl)
            t_crawl = (time.perf_counter() - t) * 1000

            corpus_version = crawl_report.corpus_version or self.store.compute_corpus_version()
            resp.corpus_version = corpus_version

            if self.store.page_count() == 0:
                resp.warnings.append("corpus is empty: crawl failed or was blocked")
                resp.timings_ms = {"total": round((time.perf_counter() - t0) * 1000, 1)}
                return resp

            # Decode the cursor before doing any work it might invalidate.
            cursor: Cursor | None = None
            if cursor_token:
                try:
                    cursor = Cursor.decode(cursor_token)
                except InvalidCursor as exc:
                    resp.warnings.append(f"ignoring cursor: {exc}")
                    cursor = None
                else:
                    if cursor.query_key != qkey:
                        resp.warnings.append("cursor belongs to a different query; showing page 1")
                        cursor = None
                    elif cursor.corpus_version != corpus_version:
                        resp.cursor_stale = True
                        resp.warnings.append(
                            "index changed since this cursor was issued; showing page 1")
                        cursor = None

            t = time.perf_counter()
            embeddings, index_cache = self.ensure_index(corpus_version)
            t_index = (time.perf_counter() - t) * 1000

            snapshot = self.store.get_query_snapshot(
                qkey, corpus_version, self.settings.query_cache_ttl_seconds)
            query_cache_hit = snapshot is not None
            t_retrieve = t_rerank = 0.0

            if snapshot is None:
                t = time.perf_counter()
                lex_hits = self.lexical.search(raw_query, self.settings.lexical_candidates)
                sem_hits = embeddings.search_pages(raw_query, self.settings.semantic_candidates)
                candidates = reciprocal_rank_fusion(lex_hits, sem_hits)
                t_retrieve = (time.perf_counter() - t) * 1000

                t = time.perf_counter()
                scored = Reranker(self.store, embeddings).rerank(raw_query, candidates)
                rows = [s.as_row() for s in scored]
                t_rerank = (time.perf_counter() - t) * 1000

                self.store.put_query_snapshot(qkey, normalized, corpus_version, rows)
            else:
                rows = snapshot["results"]

            rows.sort(key=sort_key)
            page, start = page_after(rows, cursor, page_size)

            resp.results = page
            resp.total_results = len(rows)
            resp.offset = start
            resp.has_prev = start > 0

            if page and start + len(page) < len(rows):
                last = page[-1]
                resp.next_cursor = Cursor(
                    query_key=qkey, corpus_version=corpus_version,
                    last_score=float(last["score"]), last_page_id=int(last["page_id"]),
                    offset=start + len(page),
                ).encode()

            if start > 0:
                prev_start = max(0, start - page_size)
                if prev_start > 0:
                    anchor = rows[prev_start - 1]
                    resp.prev_cursor = Cursor(
                        query_key=qkey, corpus_version=corpus_version,
                        last_score=float(anchor["score"]), last_page_id=int(anchor["page_id"]),
                        offset=prev_start,
                    ).encode()
                # prev_start == 0 means "page 1", which needs no cursor at all.

            resp.cache = {
                "crawl_skipped": crawl_report.skipped_entirely,
                "index": index_cache,
                "query_snapshot": "hit" if query_cache_hit else "miss",
                "corpus_version": corpus_version,
            }
            resp.corpus = {
                "pages": self.store.page_count(),
                "chunks": embeddings.n_chunks,
                "oldest_fetched_at": self.store.oldest_fetched_at(),
                "crawl": crawl_report.as_dict(),
            }
            resp.timings_ms = {
                "crawl": round(t_crawl, 1),
                "index": round(t_index, 1),
                "retrieve": round(t_retrieve, 1),
                "rerank": round(t_rerank, 1),
                "total": round((time.perf_counter() - t0) * 1000, 1),
            }
            return resp

    # ---------------- admin ------------------------------------------------
    def refresh(self, force: bool = True) -> dict:
        report = self.crawler.crawl(force=force)
        # Content may have changed -> ranked snapshots are no longer valid.
        self.store.purge_query_cache()
        self.ensure_index(report.corpus_version)
        return report.as_dict()

    def status(self) -> dict:
        fresh, why = self.crawler.corpus_is_fresh()
        version = self.store.compute_corpus_version()
        cached = (Path(self.settings.cache_dir) / f"lsa-{version}.npz").exists()
        return {
            "pages": self.store.page_count(),
            "chunks": len(self.store.iter_chunks()),
            "corpus_version": version,
            "corpus_fresh": fresh,
            "freshness_note": why,
            "index_artifact_cached": cached,
            "seed_urls": list(self.settings.seed_urls),
            "max_pages": self.settings.max_pages,
            "oldest_fetched_at": self.store.oldest_fetched_at(),
        }
