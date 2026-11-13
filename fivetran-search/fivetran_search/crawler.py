"""Synchronous breadth-first crawler with a hard page budget.

Frontier is a priority queue ordered by (depth, path_priority, url) so the
50-page budget is spent on shallow, high-value pages before it wanders into
deep blog archives. Pages already in the store are revalidated rather than
refetched, and a 304 reuses the persisted link graph so BFS can keep expanding
without re-parsing anything.
"""
from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field

from .config import SETTINGS, path_priority
from .extract import parse
from .fetcher import Fetcher
from .store import Store
from .urls import in_scope, normalize, path_of


@dataclass
class CrawlReport:
    pages_in_corpus: int = 0
    indexed: int = 0            # pages that counted against the page budget
    requests_made: int = 0
    frontier_remaining: int = 0
    visited: int = 0
    newly_stored: int = 0
    changed: int = 0
    revalidated: int = 0
    skipped_fresh: int = 0
    errors: int = 0
    robots_blocked: int = 0
    elapsed_s: float = 0.0
    corpus_version: str = ""
    skipped_entirely: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d["notes"] = list(self.notes)
        return d


class Crawler:
    def __init__(self, store: Store, fetcher: Fetcher | None = None):
        self.store = store
        self.fetcher = fetcher or Fetcher()

    # ---- cache-aside gate -------------------------------------------------
    def corpus_is_fresh(self) -> tuple[bool, str]:
        """Is the stored corpus good enough to serve queries without crawling?"""
        count = self.store.page_count()
        if count < min(SETTINGS.min_corpus_pages, SETTINGS.max_pages):
            return False, f"only {count} pages stored"
        oldest = self.store.oldest_fetched_at()
        if oldest is None:
            return False, "no fetch timestamps"
        age = time.time() - oldest
        if age > SETTINGS.corpus_ttl_seconds:
            return False, f"oldest page is {age / 3600:.1f}h old"
        return True, f"{count} pages, oldest {age / 3600:.1f}h old"

    def ensure_corpus(self, force: bool = False) -> CrawlReport:
        """Crawl only when needed. This is the method the query path calls."""
        fresh, why = self.corpus_is_fresh()
        if fresh and not force:
            report = CrawlReport(
                pages_in_corpus=self.store.page_count(),
                corpus_version=self.store.compute_corpus_version(),
                skipped_entirely=True,
            )
            report.notes.append(f"crawl skipped: {why}")
            return report
        return self.crawl(force=force)

    # ---- the crawl --------------------------------------------------------
    def crawl(self, force: bool = False, max_pages: int | None = None) -> CrawlReport:
        budget = max_pages or SETTINGS.max_pages
        # The budget counts pages that end up in the corpus. A 404, a redirect to
        # a non-HTML asset or a robots-blocked URL must not spend one of the 50
        # slots. A separate request cap keeps a pathological site from looping.
        request_cap = max(budget * 4, budget + 20)
        started = time.monotonic()
        report = CrawlReport()
        indexed = 0
        requests_made = 0

        # A parser change invalidates stored text even when the site is byte
        # identical, so validators must not be sent: a 304 would hand back the
        # old extraction.
        stale_parser = self.store.kv_get("parser_version") != SETTINGS.parser_version
        skip_validators = force or stale_parser
        if stale_parser:
            report.notes.append(
                f"parser v{SETTINGS.parser_version}: refetching without validators")

        frontier: list[tuple[int, int, str, str]] = []
        enqueued: set[str] = set()
        counter = 0

        def push(url: str, depth: int) -> None:
            nonlocal counter
            if url in enqueued or depth > SETTINGS.max_depth:
                return
            enqueued.add(url)
            counter += 1
            heapq.heappush(frontier, (depth, path_priority(path_of(url)), url, url))

        for seed in SETTINGS.seed_urls:
            norm = normalize(seed)
            if norm:
                push(norm, 0)

        visited: set[str] = set()

        while frontier and indexed < budget and requests_made < request_cap:
            depth, _prio, url, _tie = heapq.heappop(frontier)
            if url in visited or not in_scope(url):
                continue
            visited.add(url)

            etag, last_modified, fetched_at = self.store.get_cache_headers(url)

            # Layer 2: a page fetched within its TTL is not even revalidated.
            if (not skip_validators and fetched_at is not None
                    and time.time() - fetched_at < SETTINGS.page_ttl_seconds
                    and self.store.has_page(url)):
                report.skipped_fresh += 1
                indexed += 1
                for link in self.store.get_outlinks(url):
                    push(link, depth + 1)
                continue

            requests_made += 1
            result = self.fetcher.fetch(
                url,
                etag=None if skip_validators else etag,
                last_modified=None if skip_validators else last_modified,
            )

            if result.from_cache:
                # Layer 1: 304 Not Modified. Content is unchanged; reuse the
                # stored link graph to keep expanding the frontier.
                self.store.touch_fetched(url, 304)
                report.revalidated += 1
                indexed += 1
                for link in self.store.get_outlinks(url):
                    push(link, depth + 1)
                continue

            if result.error or not result.body:
                if result.error == "blocked by robots.txt":
                    report.robots_blocked += 1
                else:
                    report.errors += 1
                    if len(report.notes) < 15:
                        report.notes.append(f"{url}: {result.error}")
                continue

            doc = parse(url, result.body)
            existed = self.store.has_page(url)
            page_id, changed = self.store.upsert_page(
                url=url, depth=depth, http_status=result.status, etag=result.etag,
                last_modified=result.last_modified, content_hash=doc.content_hash,
                title=doc.title, meta_desc=doc.meta_desc, headings=doc.headings,
                body=doc.body, out_degree=len(doc.outlinks), ok=True,
            )
            self.store.replace_links(page_id, doc.outlinks)
            # Chunks must also be rebuilt when only the parser changed: a page
            # whose extracted text happens to be identical would otherwise keep
            # chunks produced by the previous chunking rule.
            if changed or stale_parser:
                self.store.replace_chunks(page_id, doc.chunks)
            if changed:
                report.changed += 1
            if not existed:
                report.newly_stored += 1
            indexed += 1

            for link in doc.outlinks:
                push(link, depth + 1)

        self.store.kv_set("parser_version", SETTINGS.parser_version)
        report.indexed = indexed
        report.requests_made = requests_made
        report.frontier_remaining = len(frontier)
        report.visited = len(visited)
        report.pages_in_corpus = self.store.page_count()
        report.elapsed_s = round(time.monotonic() - started, 2)
        report.corpus_version = self.store.compute_corpus_version()
        report.notes.append(
            f"http: {self.fetcher.stats['fetched']} fetched, "
            f"{self.fetcher.stats['revalidated_304']} not-modified, "
            f"{self.fetcher.stats['errors']} errors"
        )
        return report
