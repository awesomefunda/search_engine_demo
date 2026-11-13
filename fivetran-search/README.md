# Fivetran Site Search — Technical Overview

A Python MVP for keyword and natural-language search over a bounded crawl of
`fivetran.com`. The browser UI returns ranked links with snippets, explains the
ranking features, and supports cursor-based navigation with 10 results per page.

[Hosted demo](https://search-engine-demo.vercel.app/) ·
[Repository overview](../README.md) · [Deployment guide](DEPLOYMENT.md)

> The hosted application builds successfully. The latest runtime crawl reported
> `robots_blocked: 1` for its starting URL and indexed zero pages, so hosted
> search currently has no corpus. A permitted successful crawl or provisioned
> corpus is required to demonstrate results.

## Quick start

From this directory, using Python 3.12:

```sh
python -m pip install -r requirements.txt
python -m fivetran_search.cli crawl
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. Crawling requires internet access and permission
under the site's robots policy. Review the crawl report to confirm pages were
indexed. The first search also attempts a crawl if the corpus is missing or
stale; subsequent searches reuse a sufficiently complete, fresh corpus.

CLI examples:

```sh
python -m fivetran_search.cli status
python -m fivetran_search.cli search "salesforce connector" --features
python -m fivetran_search.cli search "data pipeline" --page 2
```

## Request flow

1. **Ensure the corpus:** reuse fresh pages, otherwise run the bounded crawler.
2. **Ensure the semantic index:** reuse memory or disk artifacts when the corpus
   version matches; rebuild only when needed.
3. **Check the query cache:** reuse a ranked snapshot for the normalized query
   and corpus version within its TTL.
4. **Retrieve candidates:** combine field-weighted BM25 with chunk-level LSA
   similarity, then use reciprocal rank fusion to select up to 50 candidates.
5. **Rerank and deduplicate:** score relevance features and collapse results
   sharing the same extracted-text hash.
6. **Paginate:** return rows after the cursor's score and document-ID key.

## Crawling and extraction

The crawler starts at `https://www.fivetran.com/` and recursively follows
in-scope links, preferring shallow, useful paths. Defaults limit traversal to
50 indexed pages, depth 4, and a separate request budget. Failed requests and
non-HTML responses do not consume an indexed-page slot.

URL normalization removes fragments and tracking parameters, canonicalizes the
host, and reduces duplicate URLs. Allow/deny patterns exclude assets, selected
locale paths, and routes such as login or search. The fetcher checks robots
rules, applies a minimum request delay, uses timeouts, and retries transient
connection failures. Links are extracted before navigation boilerplate is
removed from searchable text.

## Retrieval and ranking choices

| Component | Implementation and rationale | Tradeoff |
|---|---|---|
| Document store and keyword index | SQLite + FTS5: one embedded database for a small corpus, BM25, and cached snapshots | Local storage; no shared serverless persistence |
| Keyword retrieval | Field-weighted BM25 with stemming; title and headings receive higher weights | Vocabulary-dependent |
| Semantic retrieval | TF-IDF + randomized truncated SVD/LSA over overlapping text chunks; best chunk score represents a page | Relations come only from this small corpus |
| Candidate fusion | Reciprocal rank fusion combines retrieval ranks rather than incompatible raw score scales | Fusion is a candidate selection stage |
| Final ranking | Inspectable weighted features over the fused candidate set | Hand-set weights need relevance evaluation |

LSA uses up to 96 dimensions and approximately 110-word chunks with 30-word
overlap. It needs no external inference API or downloaded model. It can capture
corpus-specific word associations, but sentence queries are not guaranteed to
match when their vocabulary is absent from the corpus.

The final score uses the following configured weights:

| Feature | Weight |
|---|---:|
| Normalized keyword score | 0.30 |
| Best-chunk semantic similarity | 0.26 |
| Exact phrase match | 0.16 |
| Title and heading overlap | 0.12 |
| Query-term coverage | 0.10 |
| Term proximity | 0.04 |
| Link/depth authority | 0.02 |

Each result exposes its feature values in **why this rank**. Duplicate collapsing
uses extracted-text hashes; it identifies identical extracted content rather
than general fuzzy similarity.

## Caching and invalidation

| Layer | Reuse condition | Work avoided |
|---|---|---|
| HTTP revalidation | ETag or Last-Modified returns 304 | Downloading and parsing unchanged content |
| Page freshness | Page remains within its TTL | Fetching that page |
| Corpus freshness | Sufficient page count and fresh fetch timestamps | A complete crawl |
| Semantic artifacts | Matching corpus version, in memory or on disk | Rebuilding the semantic index |
| Query snapshots | Matching normalized query and corpus version within the cache TTL | Retrieval and reranking |

Defaults are 24 hours for page/corpus freshness and 15 minutes for query
snapshots. A content fingerprint versions the corpus, so fetching unchanged
content does not invalidate downstream caches. A parser version prompts a
refetch when extraction logic changes.

## Cursor pagination

A ranked snapshot is ordered by descending score and ascending document ID to
break ties deterministically. The cursor encodes the query key, corpus version,
and last result's sort key, plus a navigation offset. The API returns
`next_cursor` and `prev_cursor`; the browser defaults to 10 results per page.

Cursor requests skip crawling. A cursor for another query restarts at page one;
a changed corpus reports `cursor_stale` and restarts. Expired snapshots can be
recomputed. Cursors are encoded, not signed. Continuity depends on access to
the same corpus and snapshots, which temporary serverless storage cannot
provide reliably across instances.

## API

| Endpoint | Purpose |
|---|---|
| `GET /` | Browser UI |
| `GET /api/search?q=...&cursor=...&page_size=10` | Ranked results and pagination cursors |
| `GET /api/status` | Corpus size, freshness, version, and index state |
| `POST /api/refresh?force=true` | Recrawl and refresh search artifacts |
| `GET /docs` | Interactive API documentation |

Search responses include results, cursors, cache information, warnings, and
stage timings. The refresh endpoint is intended for demo administration and
requires access controls or removal for a public production service.

## Code organization

| Module | Responsibility |
|---|---|
| `app.py`, `fivetran_search/api.py` | Vercel entrypoint, FastAPI routes, and UI serving |
| `config.py`, `urls.py` | Settings, scope, normalization, and URL patterns |
| `fetcher.py`, `crawler.py`, `extract.py` | HTTP fetching, traversal, parsing, and chunking |
| `store.py` | SQLite repository and persistence |
| `lexical.py`, `semantic.py`, `fuse.py`, `rerank.py` | Retrieval and relevance pipeline |
| `pagination.py`, `pipeline.py` | Cursor handling and search orchestration |
| `cli.py`, `static/index.html` | CLI and browser UI |

Module paths in the table are relative to `fivetran_search/` unless explicitly
qualified. The design separates persistence through a repository, orchestrates
queries through a service facade, and uses cache-aside checks around expensive
stages. Retrieval, fusion, and reranking remain separate to support iteration.

## Validation

```sh
python -m unittest discover -s tests -v
```

All 33 offline tests pass in the local validation run. They cover URL handling,
HTML extraction, chunking, keyword expressions, semantic matching on a synthetic
corpus, fusion, deduplication, cache reuse, artifact round trips, and cursor
navigation including ties, query mismatches, and stale versions. These tests do
not verify live site availability, hosted crawl permission, or relevance across
arbitrary queries.

## Deployment and limitations

Vercel must use **Root Directory: `fivetran-search`** and the FastAPI preset.
The root `app.py` exports the application, while runtime SQLite and artifacts use
temporary storage when `VERCEL=1`. See [DEPLOYMENT.md](DEPLOYMENT.md).

- The live crawler currently reports a robots-policy block at the seed URL.
  Crawling must respect site permissions; the application does not bypass them.
- Crawls are synchronous. A cold search can be slow or exceed function limits.
- Temporary files are instance-local and disappear on recycling. Durable shared
  storage is needed for reliable hosted cache reuse and pagination.
- Only the bounded corpus is searchable. Client-rendered content may be missed.
- Ranking uses corpus-bound LSA and heuristic weights without a labeled
  relevance benchmark. An embedding model and evaluated reranker are potential
  next steps, not current capabilities.
- The MVP is designed for a small corpus and a single writable store. Background
  crawl jobs, shared persistence, operational monitoring, and admin access
  controls are the next steps for a production deployment.

This independent demonstration project is not affiliated with Fivetran.
