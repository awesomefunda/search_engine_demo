"""Repository layer: the one place that knows SQL.

Everything above this module talks about pages and chunks, never about tables.
This is also where the persistence half of the caching story lives:
  * pages.fetched_at / etag / last_modified  -> HTTP + page-level cache
  * pages.content_hash                      -> change detection
  * kv['corpus_version']                    -> index invalidation key
  * query_cache                             -> ranked-result snapshots for pagination
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS pages (
    id            INTEGER PRIMARY KEY,
    url           TEXT NOT NULL UNIQUE,
    depth         INTEGER NOT NULL DEFAULT 0,
    http_status   INTEGER,
    etag          TEXT,
    last_modified TEXT,
    content_hash  TEXT,
    title         TEXT DEFAULT '',
    meta_desc     TEXT DEFAULT '',
    headings      TEXT DEFAULT '',
    body          TEXT DEFAULT '',
    out_degree    INTEGER DEFAULT 0,
    fetched_at    REAL,
    ok            INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_pages_fetched ON pages(fetched_at);

CREATE TABLE IF NOT EXISTS links (
    src_id  INTEGER NOT NULL,
    dst_url TEXT NOT NULL,
    PRIMARY KEY (src_id, dst_url)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    page_id INTEGER NOT NULL,
    ord     INTEGER NOT NULL,
    text    TEXT NOT NULL,
    UNIQUE (page_id, ord)
);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(page_id);

-- Lexical index. Separate field columns let bm25() apply per-field weights.
-- Deliberately NOT an external-content table: those require the previous
-- column values to be replayed on delete, and a plain table lets a page be
-- reindexed with DELETE + INSERT on its rowid.
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    title, headings, meta_desc, body, url, tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS kv (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS query_cache (
    qkey           TEXT PRIMARY KEY,
    query          TEXT NOT NULL,
    corpus_version TEXT NOT NULL,
    created_at     REAL NOT NULL,
    total          INTEGER NOT NULL,
    results_json   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qc_created ON query_cache(created_at);
"""


@dataclass
class PageRow:
    id: int
    url: str
    title: str
    meta_desc: str
    headings: str
    body: str
    depth: int
    out_degree: int
    content_hash: str
    fetched_at: float


class Store:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ---------------- generic kv -------------------------------------------
    def kv_get(self, key: str, default: Any = None) -> Any:
        row = self.conn.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def kv_set(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO kv(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )
        self.conn.commit()

    # ---------------- crawl bookkeeping ------------------------------------
    def get_cache_headers(self, url: str) -> tuple[str | None, str | None, float | None]:
        row = self.conn.execute(
            "SELECT etag, last_modified, fetched_at FROM pages WHERE url=?", (url,)
        ).fetchone()
        if not row:
            return None, None, None
        return row["etag"], row["last_modified"], row["fetched_at"]

    def has_page(self, url: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM pages WHERE url=? AND ok=1", (url,)
        ).fetchone() is not None

    def touch_fetched(self, url: str, status: int) -> None:
        """Mark a page verified-fresh without rewriting content (the 304 path)."""
        self.conn.execute(
            "UPDATE pages SET fetched_at=?, http_status=? WHERE url=?",
            (time.time(), status, url),
        )
        self.conn.commit()

    def get_outlinks(self, url: str) -> list[str]:
        rows = self.conn.execute(
            "SELECT l.dst_url FROM links l JOIN pages p ON p.id = l.src_id WHERE p.url=?",
            (url,),
        ).fetchall()
        return [r["dst_url"] for r in rows]

    def upsert_page(self, *, url: str, depth: int, http_status: int, etag: str | None,
                    last_modified: str | None, content_hash: str, title: str, meta_desc: str,
                    headings: str, body: str, out_degree: int, ok: bool) -> tuple[int, bool]:
        """Insert or update a page. Returns (page_id, content_changed)."""
        prev = self.conn.execute(
            "SELECT id, content_hash FROM pages WHERE url=?", (url,)
        ).fetchone()
        now = time.time()
        if prev is None:
            cur = self.conn.execute(
                "INSERT INTO pages(url,depth,http_status,etag,last_modified,content_hash,title,"
                "meta_desc,headings,body,out_degree,fetched_at,ok) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (url, depth, http_status, etag, last_modified, content_hash, title, meta_desc,
                 headings, body, out_degree, now, int(ok)),
            )
            pid = int(cur.lastrowid)
            changed = True
        else:
            pid = int(prev["id"])
            changed = prev["content_hash"] != content_hash
            self.conn.execute(
                "UPDATE pages SET depth=MIN(depth,?), http_status=?, etag=?, last_modified=?, "
                "content_hash=?, title=?, meta_desc=?, headings=?, body=?, out_degree=?, "
                "fetched_at=?, ok=? WHERE id=?",
                (depth, http_status, etag, last_modified, content_hash, title, meta_desc,
                 headings, body, out_degree, now, int(ok), pid),
            )
        self._reindex_fts(pid)
        self.conn.commit()
        return pid, changed

    def _reindex_fts(self, page_id: int) -> None:
        self.conn.execute("DELETE FROM pages_fts WHERE rowid=?", (page_id,))
        self.conn.execute(
            "INSERT INTO pages_fts(rowid, title, headings, meta_desc, body, url) "
            "SELECT id, title, headings, meta_desc, body, url FROM pages WHERE id=?",
            (page_id,),
        )

    def replace_links(self, src_id: int, dst_urls: Iterable[str]) -> None:
        self.conn.execute("DELETE FROM links WHERE src_id=?", (src_id,))
        self.conn.executemany(
            "INSERT OR IGNORE INTO links(src_id,dst_url) VALUES(?,?)",
            [(src_id, u) for u in dst_urls],
        )
        self.conn.commit()

    def replace_chunks(self, page_id: int, texts: Sequence[str]) -> None:
        self.conn.execute("DELETE FROM chunks WHERE page_id=?", (page_id,))
        self.conn.executemany(
            "INSERT INTO chunks(page_id, ord, text) VALUES(?,?,?)",
            [(page_id, i, t) for i, t in enumerate(texts)],
        )
        self.conn.commit()

    # ---------------- corpus reads ----------------------------------------
    def page_count(self, only_ok: bool = True) -> int:
        q = "SELECT COUNT(*) c FROM pages"
        if only_ok:
            q += " WHERE ok=1"
        return int(self.conn.execute(q).fetchone()["c"])

    def oldest_fetched_at(self) -> float | None:
        row = self.conn.execute("SELECT MIN(fetched_at) m FROM pages WHERE ok=1").fetchone()
        return row["m"]

    PAGE_COLS = ("SELECT id,url,title,meta_desc,headings,body,depth,out_degree,"
                 "COALESCE(content_hash,'') content_hash, COALESCE(fetched_at,0) fetched_at "
                 "FROM pages")

    def iter_pages(self) -> list[PageRow]:
        rows = self.conn.execute(self.PAGE_COLS + " WHERE ok=1 ORDER BY id").fetchall()
        return [PageRow(**dict(r)) for r in rows]

    def get_pages_by_ids(self, ids: Sequence[int]) -> dict[int, PageRow]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.conn.execute(
            self.PAGE_COLS + " WHERE id IN (" + marks + ")", tuple(ids)
        ).fetchall()
        return {int(r["id"]): PageRow(**dict(r)) for r in rows}

    def iter_chunks(self) -> list[tuple[int, int, str]]:
        rows = self.conn.execute(
            "SELECT c.id, c.page_id, c.text FROM chunks c JOIN pages p ON p.id=c.page_id "
            "WHERE p.ok=1 ORDER BY c.page_id, c.ord"
        ).fetchall()
        return [(int(r["id"]), int(r["page_id"]), r["text"]) for r in rows]

    def chunks_for_page(self, page_id: int) -> list[str]:
        rows = self.conn.execute(
            "SELECT text FROM chunks WHERE page_id=? ORDER BY ord", (page_id,)
        ).fetchall()
        return [r["text"] for r in rows]

    def inlink_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT dst_url, COUNT(*) c FROM links GROUP BY dst_url"
        ).fetchall()
        return {r["dst_url"]: int(r["c"]) for r in rows}

    # ---------------- corpus version (index cache key) ---------------------
    def compute_corpus_version(self) -> str:
        """Content-addressed fingerprint of the corpus.

        Changes only when a page is added, removed, or its content actually
        changed. A recrawl that finds nothing new leaves this identical, so
        cached index artifacts and query snapshots stay valid.
        """
        rows = self.conn.execute(
            "SELECT url, COALESCE(content_hash,'') h FROM pages WHERE ok=1 ORDER BY url"
        ).fetchall()
        h = hashlib.blake2b(digest_size=16)
        for r in rows:
            h.update(r["url"].encode())
            h.update(b"\x00")
            h.update(r["h"].encode())
            h.update(b"\x00")
        return h.hexdigest()

    # ---------------- query snapshot cache --------------------------------
    def get_query_snapshot(self, qkey: str, corpus_version: str, ttl: float) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM query_cache WHERE qkey=? AND corpus_version=?",
            (qkey, corpus_version),
        ).fetchone()
        if not row:
            return None
        if time.time() - row["created_at"] > ttl:
            return None
        return {
            "total": int(row["total"]),
            "results": json.loads(row["results_json"]),
            "created_at": row["created_at"],
        }

    def put_query_snapshot(self, qkey: str, query: str, corpus_version: str,
                           results: list[dict]) -> None:
        self.conn.execute(
            "INSERT INTO query_cache(qkey,query,corpus_version,created_at,total,results_json) "
            "VALUES(?,?,?,?,?,?) ON CONFLICT(qkey) DO UPDATE SET "
            "corpus_version=excluded.corpus_version, created_at=excluded.created_at, "
            "total=excluded.total, results_json=excluded.results_json",
            (qkey, query, corpus_version, time.time(), len(results), json.dumps(results)),
        )
        self.conn.commit()

    def purge_query_cache(self) -> None:
        self.conn.execute("DELETE FROM query_cache")
        self.conn.commit()

    # ---------------- lexical search --------------------------------------
    def bm25_search(self, fts_query: str, limit: int) -> list[tuple[int, float]]:
        """Returns [(page_id, positive_score)] best-first.

        Field weights favour title and headings over body text.
        """
        try:
            rows = self.conn.execute(
                "SELECT rowid, bm25(pages_fts, 8.0, 3.0, 2.0, 1.0, 2.0) AS s "
                "FROM pages_fts WHERE pages_fts MATCH ? ORDER BY s LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 expression -> treat as no lexical hits.
            return []
        # bm25() returns negatives, more negative = better match. Flip the sign.
        return [(int(r["rowid"]), -float(r["s"])) for r in rows]

    def close(self) -> None:
        self.conn.close()
