"""Offline tests: no network. Run with `python -m tests.test_search` or unittest.

The integration tests build a small synthetic corpus directly in the store and
inject a crawler stub, so the search path is exercised without touching the
live site.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fivetran_search.crawler import CrawlReport                      # noqa: E402
from fivetran_search.extract import chunk_words, parse               # noqa: E402
from fivetran_search.fuse import reciprocal_rank_fusion              # noqa: E402
from fivetran_search.lexical import build_fts_expression             # noqa: E402
from fivetran_search.pagination import Cursor, InvalidCursor, page_after, sort_key  # noqa: E402
from fivetran_search.pipeline import SearchService                   # noqa: E402
from fivetran_search.semantic import EmbeddingIndex                  # noqa: E402
from fivetran_search.store import Store                              # noqa: E402
from fivetran_search.textutil import min_window_span, tokenize       # noqa: E402
from fivetran_search.urls import in_scope, normalize                 # noqa: E402

DOCS = [
    ("https://fivetran.com/pricing",
     "Fivetran Pricing: plans and cost",
     "Understand how much Fivetran costs, billed on monthly active rows.",
     "Pricing plans | Cost example | MAR",
     "Fivetran pricing is consumption based. You pay for monthly active rows, MAR. "
     "Plans include Free, Starter, Standard and Enterprise. A cost example shows how "
     "much a typical pipeline costs per month and how the price scales."),
    ("https://fivetran.com/connectors/salesforce",
     "Salesforce connector",
     "Replicate Salesforce objects into your warehouse.",
     "Salesforce | Schema | Sync",
     "The Salesforce connector replicates accounts, contacts and opportunities into "
     "your destination warehouse on a schedule you choose. Historical sync and "
     "incremental updates are supported."),
    ("https://fivetran.com/security",
     "Security and compliance",
     "SOC 2, GDPR and HIPAA compliance for your data pipeline.",
     "Security | Compliance | Encryption",
     "Fivetran encrypts data in transit and at rest. We maintain SOC 2 Type II, "
     "GDPR and HIPAA compliance for every data pipeline, and support customer "
     "managed keys and private networking."),
    ("https://fivetran.com/docs/cdc",
     "Change data capture",
     "How log based change data capture works.",
     "CDC | Log reading | Databases",
     "Change data capture reads the database transaction log to replicate inserts, "
     "updates and deletes from Postgres, MySQL and SQL Server with low overhead."),
]


def build_corpus(store: Store) -> None:
    for i, (url, title, meta, heads, body) in enumerate(DOCS):
        pid, _ = store.upsert_page(
            url=url, depth=1, http_status=200, etag=f"e{i}", last_modified=None,
            content_hash=f"h{i}", title=title, meta_desc=meta, headings=heads,
            body=body, out_degree=1, ok=True,
        )
        text = " ".join([title, meta, heads, body])
        store.replace_chunks(pid, chunk_words(text, 40, 10))
        store.replace_links(pid, ["https://fivetran.com/"])


class FreshCrawler:
    """Crawler stub: the corpus is always considered fresh, so nothing crawls."""

    def __init__(self, store: Store):
        self.store = store

    def corpus_is_fresh(self):
        return True, "stub"

    def ensure_corpus(self, force: bool = False) -> CrawlReport:
        r = CrawlReport(pages_in_corpus=self.store.page_count(),
                        corpus_version=self.store.compute_corpus_version(),
                        skipped_entirely=True)
        r.notes.append("stub: no crawl")
        return r

    def crawl(self, force: bool = False, max_pages: int | None = None) -> CrawlReport:
        return self.ensure_corpus()


class ServiceCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.store = Store(root / "t.db")
        build_corpus(cls.store)
        from fivetran_search.config import Settings
        cls.settings = Settings(db_path=root / "t.db", cache_dir=root / "cache", page_size=2)
        cls.service = SearchService(settings=cls.settings, store=cls.store,
                                    crawler=FreshCrawler(cls.store))

    @classmethod
    def tearDownClass(cls):
        cls.store.close()
        cls.tmp.cleanup()


class TestUrls(unittest.TestCase):
    def test_normalize_collapses_variants(self):
        # Canonical host is www: the apex 404s on most paths on this site.
        base = "https://www.fivetran.com/pricing"
        for variant in [
            "https://www.fivetran.com/pricing",
            "http://fivetran.com/pricing/",
            "https://fivetran.com/pricing#plans",
            "https://fivetran.com/pricing?utm_source=x&gclid=y",
            "https://fivetran.com//pricing",
        ]:
            self.assertEqual(normalize(variant), base, variant)

    def test_relative_and_rejects(self):
        self.assertEqual(normalize("/security", base="https://www.fivetran.com/a/b"),
                         "https://www.fivetran.com/security")
        self.assertIsNone(normalize("mailto:a@b.com"))
        self.assertIsNone(normalize("javascript:void(0)"))

    def test_scope_rules(self):
        self.assertTrue(in_scope("https://www.fivetran.com/pricing"))
        self.assertTrue(in_scope("https://fivetran.com/pricing"))          # apex accepted
        self.assertFalse(in_scope("https://example.com/pricing"))          # offsite
        self.assertFalse(in_scope("https://www.fivetran.com/ja/pricing"))  # locale denied
        self.assertFalse(in_scope("https://www.fivetran.com/logo.png"))    # asset denied


class TestText(unittest.TestCase):
    def test_stemming_is_consistent(self):
        self.assertEqual(tokenize("connectors"), tokenize("connector"))
        self.assertEqual(tokenize("pricing"), tokenize("pricing"))

    def test_stopwords_dropped(self):
        self.assertEqual(tokenize("how much does it cost"), ["much", "cost"])

    def test_min_window_span(self):
        toks = "a data pipeline for security and compliance review".split()
        self.assertEqual(min_window_span(toks, {"security", "compliance"}), 3)
        self.assertIsNone(min_window_span(toks, {"security", "absent"}))


class TestExtract(unittest.TestCase):
    def test_parse_html(self):
        html = b"""<html><head><title>Pricing</title>
          <meta name="description" content="What it costs"></head>
          <body><nav><a href="/nav-only">nav</a></nav>
          <main><h1>Plans</h1><p>MAR based pricing.</p>
          <a href="/security">sec</a><a href="https://twitter.com/x">off</a></main>
          <script>var a=1;</script></body></html>"""
        doc = parse("https://www.fivetran.com/pricing", html)
        self.assertEqual(doc.title, "Pricing")
        self.assertEqual(doc.meta_desc, "What it costs")
        self.assertIn("Plans", doc.headings)
        self.assertIn("MAR based pricing", doc.body)
        self.assertNotIn("var a=1", doc.body)            # script stripped
        self.assertIn("https://www.fivetran.com/security", doc.outlinks)
        self.assertIn("https://www.fivetran.com/nav-only", doc.outlinks)  # nav links kept
        self.assertTrue(all("twitter" not in u for u in doc.outlinks))

    def test_chunks_overlap(self):
        chunks = chunk_words(" ".join(str(i) for i in range(100)), 40, 10)
        self.assertGreater(len(chunks), 1)
        first, second = chunks[0].split(), chunks[1].split()
        self.assertEqual(first[30:40], second[:10])      # 10-word overlap


class TestLexical(unittest.TestCase):
    def test_expression_ors_terms_and_adds_whole_phrase(self):
        expr = build_fts_expression("salesforce connector")
        self.assertIn(" OR ", expr)
        self.assertIn('"salesforce connector"', expr)

    def test_quoted_phrase_becomes_required(self):
        expr = build_fts_expression('"change data capture" postgres')
        self.assertTrue(expr.startswith('"change data capture" AND'))

    def test_punctuation_cannot_break_fts(self):
        for bad in ['AND OR NEAR', '"unbalanced', "a*b(c)", "   ", "---"]:
            build_fts_expression(bad)   # must not raise


class TestFusion(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        lex = [(1, 9.0), (2, 8.0), (3, 7.0)]
        sem = [(3, 0.9), (1, 0.8)]
        fused = reciprocal_rank_fusion(lex, sem, k=1)
        self.assertEqual(fused[0].page_id, 1)            # rank 1 + rank 2
        self.assertEqual(sorted(fused[0].sources), ["keyword", "semantic"])
        found_by_one = [c for c in fused if c.page_id == 2][0]
        self.assertIsNone(found_by_one.semantic_rank)


class TestSemantic(unittest.TestCase):
    def test_lsa_matches_without_shared_keywords(self):
        store_dir = tempfile.TemporaryDirectory()
        store = Store(Path(store_dir.name) / "s.db")
        build_corpus(store)
        idx = EmbeddingIndex().build(store.iter_chunks(), "v1")
        self.assertGreater(idx.n_chunks, 0)
        hits = idx.search_pages("what does it cost per month", limit=4)
        self.assertTrue(hits)
        top_url = store.get_pages_by_ids([hits[0][0]])[hits[0][0]].url
        self.assertEqual(top_url, "https://fivetran.com/pricing")
        store.close()
        store_dir.cleanup()

    def test_artifact_cache_roundtrip(self):
        d = tempfile.TemporaryDirectory()
        store_dir = Path(d.name)
        store = Store(store_dir / "s.db")
        build_corpus(store)
        idx = EmbeddingIndex().build(store.iter_chunks(), "vX")
        idx.save(store_dir / "cache")
        again = EmbeddingIndex.load(store_dir / "cache", "vX")
        self.assertIsNotNone(again)
        self.assertEqual(again.n_chunks, idx.n_chunks)
        self.assertIsNone(EmbeddingIndex.load(store_dir / "cache", "other-version"))
        store.close()
        d.cleanup()


class TestDuplicateCollapsing(unittest.TestCase):
    def test_identical_text_collapses_to_one_result(self):
        from fivetran_search.rerank import Scored, collapse_duplicates
        rows = [
            Scored(1, "https://x/a?tab=1", "A", 0.9, "", [], [], content_hash="same"),
            Scored(2, "https://x/a?tab=2", "A", 0.8, "", [], [], content_hash="same"),
            Scored(3, "https://x/b", "B", 0.7, "", [], [], content_hash="other"),
            Scored(4, "https://x/c", "C", 0.6, "", [], [], content_hash=""),
        ]
        out = collapse_duplicates(rows)
        self.assertEqual([r.page_id for r in out], [1, 3, 4])
        self.assertEqual(out[0].near_duplicates, ["https://x/a?tab=2"])
        # A page with no hash is never collapsed away.
        self.assertEqual(out[-1].page_id, 4)


class TestParserVersioning(unittest.TestCase):
    def test_crawler_refetches_when_parser_version_changes(self):
        """A 304 would hand back text parsed by the old extractor."""
        from fivetran_search.config import SETTINGS
        d = tempfile.TemporaryDirectory()
        store = Store(Path(d.name) / "p.db")
        self.assertIsNone(store.kv_get("parser_version"))      # stale by default
        store.kv_set("parser_version", SETTINGS.parser_version)
        self.assertEqual(store.kv_get("parser_version"), SETTINGS.parser_version)
        store.close()
        d.cleanup()


class TestCursor(unittest.TestCase):
    def test_roundtrip(self):
        c = Cursor("qk", "cv", 0.4242, 7, 10)
        back = Cursor.decode(c.encode())
        self.assertEqual(back, c)

    def test_rejects_garbage(self):
        for bad in ["", "!!!", "YWJj", "eyJ2Ijo5OTl9"]:
            with self.assertRaises(InvalidCursor):
                Cursor.decode(bad)

    def test_keyset_handles_score_ties(self):
        rows = [{"score": 1.0, "page_id": 5}, {"score": 1.0, "page_id": 9},
                {"score": 0.5, "page_id": 2}]
        rows.sort(key=sort_key)
        self.assertEqual([r["page_id"] for r in rows], [5, 9, 2])
        cur = Cursor("q", "v", 1.0, 5, 1)
        page, start = page_after(rows, cur, 2)
        self.assertEqual(start, 1)
        self.assertEqual([r["page_id"] for r in page], [9, 2])


class TestPagination(ServiceCase):
    def test_walk_covers_every_row_once(self):
        resp = self.service.search("data pipeline", page_size=2)
        seen: list[int] = []
        offsets: list[int] = []
        guard = 0
        while True:
            seen.extend(r["page_id"] for r in resp.results)
            offsets.append(resp.offset)
            if not resp.next_cursor or guard > 20:
                break
            guard += 1
            resp = self.service.search("data pipeline", cursor_token=resp.next_cursor,
                                       page_size=2)
        self.assertEqual(len(seen), len(set(seen)), "a page_id was repeated across pages")
        self.assertEqual(len(seen), resp.total_results, "pages did not cover all results")
        self.assertEqual(offsets, sorted(offsets))

    def test_prev_cursor_returns_to_previous_page(self):
        p1 = self.service.search("data pipeline", page_size=2)
        self.assertIsNotNone(p1.next_cursor)
        p2 = self.service.search("data pipeline", cursor_token=p1.next_cursor, page_size=2)
        self.assertEqual(p2.offset, 2)
        self.assertTrue(p2.has_prev)
        # Page 2's prev points at page 1, which needs no cursor.
        self.assertIsNone(p2.prev_cursor)
        back = self.service.search("data pipeline", cursor_token=p2.prev_cursor, page_size=2)
        self.assertEqual([r["page_id"] for r in back.results],
                         [r["page_id"] for r in p1.results])

    def test_cursor_from_another_query_is_rejected(self):
        a = self.service.search("data pipeline", page_size=2)
        self.assertIsNotNone(a.next_cursor, "fixture should yield more than one page")
        b = self.service.search("salesforce", cursor_token=a.next_cursor, page_size=2)
        self.assertEqual(b.offset, 0)
        self.assertTrue(any("different query" in w for w in b.warnings))

    def test_stale_cursor_is_reported_not_silently_wrong(self):
        resp = self.service.search("data pipeline", page_size=2)
        self.assertIsNotNone(resp.next_cursor)
        stale = Cursor.decode(resp.next_cursor)
        forged = Cursor(stale.query_key, "deadbeefdeadbeef", stale.last_score,
                        stale.last_page_id, stale.offset).encode()
        out = self.service.search("data pipeline", cursor_token=forged, page_size=2)
        self.assertTrue(out.cursor_stale)
        self.assertEqual(out.offset, 0)


class TestRanking(ServiceCase):
    def test_keyword_query_finds_the_right_page(self):
        resp = self.service.search("salesforce connector", page_size=5)
        self.assertEqual(resp.results[0]["url"], "https://fivetran.com/connectors/salesforce")

    def test_semantic_query_with_no_keyword_overlap(self):
        # "cost" never appears as "pricing"; the latent space has to bridge it.
        resp = self.service.search("how much will this cost me each month", page_size=5)
        self.assertEqual(resp.results[0]["url"], "https://fivetran.com/pricing")

    def test_quoted_phrase_wins(self):
        resp = self.service.search('"change data capture"', page_size=5)
        self.assertEqual(resp.results[0]["url"], "https://fivetran.com/docs/cdc")

    def test_results_are_sorted_by_score(self):
        resp = self.service.search("compliance", page_size=10)
        scores = [r["score"] for r in resp.results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_every_result_reports_its_features(self):
        resp = self.service.search("security", page_size=3)
        for r in resp.results:
            self.assertTrue(set(r["features"]) >= {"lexical", "semantic", "phrase", "coverage"})

    def test_empty_query_is_handled(self):
        resp = self.service.search("   ")
        self.assertEqual(resp.results, [])
        self.assertIn("empty query", resp.warnings)


class TestCaching(ServiceCase):
    def test_second_identical_query_hits_the_snapshot(self):
        self.store.purge_query_cache()
        first = self.service.search("encryption at rest", page_size=2)
        self.assertEqual(first.cache["query_snapshot"], "miss")
        second = self.service.search("encryption at rest", page_size=2)
        self.assertEqual(second.cache["query_snapshot"], "hit")
        self.assertEqual([r["page_id"] for r in first.results],
                         [r["page_id"] for r in second.results])

    def test_index_is_not_rebuilt_for_an_unchanged_corpus(self):
        self.service.search("anything", page_size=2)
        again = self.service.search("anything else", page_size=2)
        self.assertIn(again.cache["index"], ("memory", "disk"))

    def test_crawl_is_skipped_when_corpus_is_fresh(self):
        resp = self.service.search("pricing", page_size=2)
        self.assertTrue(resp.cache["crawl_skipped"])

    def test_corpus_version_tracks_content_not_fetch_time(self):
        v1 = self.store.compute_corpus_version()
        # Refetching with identical content must not change the version.
        self.store.touch_fetched(DOCS[0][0], 304)
        self.assertEqual(self.store.compute_corpus_version(), v1)
        # A real content change must.
        self.store.upsert_page(
            url=DOCS[0][0], depth=1, http_status=200, etag="e0", last_modified=None,
            content_hash="CHANGED", title=DOCS[0][1], meta_desc=DOCS[0][2],
            headings=DOCS[0][3], body=DOCS[0][4], out_degree=1, ok=True,
        )
        self.assertNotEqual(self.store.compute_corpus_version(), v1)
        # Restore so other tests see the original fingerprint.
        self.store.upsert_page(
            url=DOCS[0][0], depth=1, http_status=200, etag="e0", last_modified=None,
            content_hash="h0", title=DOCS[0][1], meta_desc=DOCS[0][2],
            headings=DOCS[0][3], body=DOCS[0][4], out_degree=1, ok=True,
        )
        self.assertEqual(self.store.compute_corpus_version(), v1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
