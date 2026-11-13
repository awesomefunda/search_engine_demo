# Fivetran Site Search (MVP)

A lightweight internal search engine over a bounded crawl of `fivetran.com`.
Type a keyword or a whole sentence; get relevance-ordered results, 10 per page,
cursor paginated. Runs entirely locally: no API keys, no model downloads, no
services beyond Python.

```bash
pip install requests beautifulsoup4 lxml numpy fastapi uvicorn

python -m fivetran_search.cli crawl            # build the corpus (50 pages)
uvicorn fivetran_search.api:app --port 8000    # UI at http://localhost:8000
```

CLI, if you prefer it:

```bash
python -m fivetran_search.cli search "how much does it cost" --features
python -m fivetran_search.cli search "salesforce connector" --page 2
python -m fivetran_search.cli status
```

## How a query is served

```
  "how much does this cost"
          |
   SearchService.search()                    <- single facade, used by UI/API/CLI
          |
   1. ensure_corpus()        crawl ONLY if the corpus is missing or past TTL
   2. ensure_index()         rebuild LSA ONLY if corpus_version changed
   3. query snapshot?        hit  -> reuse the ranked list verbatim
          |                  miss -> 4,5,6
   4. retrieve (recall)      BM25 over FTS5   +   LSA cosine over chunks
   5. fuse                   reciprocal rank fusion -> top 50 candidates
   6. rerank (precision)     phrase / title / coverage / proximity / authority
          |
   7. keyset page            10 rows after the cursor's sort key
```

Retrieve-wide then rank-narrow is the standard split: step 4 is cheap and runs
over everything, step 6 is expensive but only ever sees ~50 documents.

## Crawling

Breadth-first from `https://www.fivetran.com/`, following links on child pages
recursively, synchronous by design.

- **Budget counts indexed pages, not requests.** A 404 or a non-HTML response
  must not spend one of the 50 slots. A separate request cap (`4x` the page
  budget) bounds total work.
- **Frontier is a priority queue** on `(depth, path_priority, url)`, so the
  budget is spent on shallow, high-value paths (`/pricing`, `/connectors`,
  `/docs`, `/security`) before it wanders into deep archives.
- **robots.txt is honored** per host via `urllib.robotparser`, including
  `Crawl-delay`, with a 0.5 s politeness floor between requests and a generic
  bot User-Agent.
- **URL normalization** collapses the ways one page can be addressed: scheme,
  host spelling, trailing slash, duplicate slashes, `#fragment`, and tracking
  params (`utm_*`, `gclid`, ...). Remaining query params are sorted so key
  order cannot create a duplicate.
- **Canonical host matters here.** `fivetran.com` and `www.fivetran.com` are the
  same site and must collapse to one host, but the apex returns **404** for most
  paths that `www` serves. Normalizing to the apex silently cost 32 of 50 pages
  before this was caught; the canonical host is configurable for exactly that
  reason.
- **Deny patterns** keep the budget on content: assets, locale trees (`/ja`,
  `/jp`), `/search`, `/login`, `/dashboard`, blog pagination.
- **Link extraction happens before boilerplate removal** (nav menus are the
  cheapest route to a site's top-level pages); text extraction happens after it,
  preferring `<main>`.

## Caching: four layers

The requirement was that a later query must not re-crawl unless it has to.
Each layer has a different invalidation key, which is the part that matters.

| Layer | Key | Invalidated by | What it saves |
|---|---|---|---|
| 1. HTTP revalidation | `ETag` / `Last-Modified` per URL | server says `200` instead of `304` | download + parse |
| 2. Page freshness | `pages.fetched_at` vs `page_ttl_seconds` | TTL expiry | the request entirely |
| 3. Corpus gate | page count + oldest `fetched_at` | `corpus_ttl_seconds` | **the whole crawl** |
| 4. Index artifacts | `corpus_version` | any real content change, or a `parser_version` bump | the SVD rebuild |
| 5. Query snapshot | `(query_key, corpus_version)` | content change or `query_cache_ttl` | retrieval + reranking |

`corpus_version` is a content-addressed fingerprint: `blake2b` over sorted
`(url, content_hash)` pairs, where `content_hash` is taken over **extracted
text**, not raw HTML. So a rotating build id or CSRF token in the markup does
not count as a change, a `304` does not change it, and an unchanged site after
a recrawl leaves every downstream cache valid.

Consequence: a cold first query crawls; every later query for the next 24 h
answers from SQLite, and page 2 of any query is one row read. Measured on the
live corpus: 937 ms cold, 17 ms on a snapshot hit, 1.5 ms once the index is also
resident in process.

**The cache can be too good.** After `extract.py` was changed to drop nav
mega-menu headings, a forced recrawl returned 46 `304 Not Modified` responses and
kept the *old* extraction: the site had not changed, but the parser had. So
`parser_version` is a separate invalidation axis. When it differs from what the
store recorded, the next crawl refetches without validators, because a 304 would
hand back text parsed by the previous version.

## Ranking

**Keyword (BM25).** SQLite FTS5 with a Porter tokenizer. Fields are separate
columns so `bm25()` can weight them: title 8, headings 3, meta 2, url 2, body 1.
Query terms are OR-ed for recall, and the full input is also offered as a phrase
so intact matches score higher. Quoted input becomes a required phrase.

**Semantic (TF-IDF + LSA).** Pages are chunked into ~110-word overlapping
windows; chunks are TF-IDF vectors reduced to 96 dimensions by a randomized
truncated SVD, and a query is projected into the same space. Chunk scores are
max-pooled to page level, so one strongly matching passage is enough.

LSA rather than a transformer because the constraint was no API key and no model
download, and it still bridges vocabulary: *"how much will this cost me"* ranks
`/pricing` first with no shared keyword, because *cost*, *pricing* and *MAR*
co-occur across the corpus. `EmbeddingIndex` is a Strategy — a
sentence-transformers backend can replace it without the query path changing.
Randomized SVD keeps a rebuild at `O(nnz * k)` instead of `O(n^2 V)`.

**Fusion (RRF).** `1/(60 + rank)` summed per retriever. Ranks, not raw scores:
BM25 output is unbounded (and SQLite's is signed), cosine is `[-1, 1]`, so the
two are not comparable on value. RRF also rewards agreement between retrievers,
which is the signal that a document is genuinely on-topic.

**Rerank.** A documented linear blend over the ~50 fused candidates:

| Feature | Weight | Signal |
|---|---|---|
| `lexical` | 0.30 | BM25, min-max normalized within the candidate set |
| `semantic` | 0.26 | best chunk cosine |
| `phrase` | 0.16 | exact phrase: 1.0 in title/headings, 0.6 in body |
| `title` | 0.12 | query-term overlap with title + headings |
| `coverage` | 0.10 | fraction of query terms present at all |
| `proximity` | 0.04 | shortest window containing every query term |
| `authority` | 0.02 | inlink count + shallow depth |

Linear and inspectable beats a learned reranker at MVP scale: every result
carries its own `features` dict, and the UI's "why this rank" exposes it. The
weights are a starting point to be tuned against judgments, not a claim of
optimality.

**Near-duplicate collapsing.** Ranked results are then deduplicated on the
extracted-text hash. Four `/industries?tab=*` URLs return byte-identical HTML
because the tabs switch client side, and without this they took four of the ten
slots on the first page. The best-scoring URL wins and the rest are reported on
it as `near_duplicates`, which the UI shows as "+3 duplicate pages collapsed".

## Cursor pagination

Offset pagination over a live ranking drifts — if the index changes between page
1 and page 2, rows shift and the user sees duplicates or gaps. Instead:

1. the full ranked list is computed once and **snapshotted** under
   `(query_key, corpus_version)`;
2. the cursor carries the **sort key of the last row served** —
   `{query_key, corpus_version, last_score, last_page_id, offset}`,
   base64url JSON;
3. the next page is a keyset scan for rows strictly after that key.

Details that make it correct rather than approximately correct:

- **Total order.** Sorting on score alone is not a total order; ties would make
  two rows mutually "after" each other. The key is `(-score, page_id)`.
- **Cursor is scoped to its query.** A cursor from another query is rejected
  rather than applied to the wrong ranking.
- **Staleness is reported, not hidden.** A cursor minted against a different
  `corpus_version` returns page 1 with `cursor_stale: true`, instead of
  silently indexing into a ranking that no longer exists.
- **Pagination never crawls.** A request carrying a cursor skips the corpus gate
  outright, so page 2 cannot be ranked against a corpus page 1 never saw.
- Backward paging works the same way (`prev_cursor`); page 1 needs no cursor, so
  `prev_cursor` is `null` there.

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/search?q=&cursor=&page_size=` | the query path |
| `GET /api/status` | corpus size, freshness, `corpus_version`, cache state |
| `POST /api/refresh?force=true` | recrawl, purge query snapshots, rebuild index |

A search response reports `results`, `total_results`, `offset`, `next_cursor`,
`prev_cursor`, `cursor_stale`, plus `cache` (which layers hit) and `timings_ms`
per stage — the UI's status bar is just that data rendered.

## Layout

```
fivetran_search/
  config.py      settings, URL allow/deny patterns, rerank weights
  urls.py        normalization + scope rules
  fetcher.py     polite HTTP, robots.txt, conditional GET, one retry
  extract.py     HTML -> fielded doc + outlinks + chunks
  crawler.py     BFS frontier, page budget, cache-aside corpus gate
  store.py       SQLite repository (pages, links, chunks, FTS5, caches)
  lexical.py     query -> FTS5 expression -> BM25
  semantic.py    TF-IDF + randomized SVD (hand-rolled CSR, no scipy)
  textutil.py    tokenizer, stemmer, min-window span
  fuse.py        reciprocal rank fusion
  rerank.py      feature scorer + snippet selection
  pagination.py  cursor encode/decode, keyset scan
  pipeline.py    SearchService facade
  api.py         FastAPI + static UI
  cli.py         crawl / search / status
  static/index.html
tests/test_search.py
```

Patterns in use: **Repository** (`Store` owns all SQL), **Facade**
(`SearchService`), **Strategy** (retrievers and the embedding backend),
**Pipeline** (retrieve -> fuse -> rerank -> paginate), **Cache-aside** (every
layer above), **Value object** (immutable `Cursor`).

## Tests

```bash
python -m tests.test_search
```

33 tests, no network: they build a synthetic corpus in a temp SQLite file and
inject a crawler stub. Covered: URL normalization and scope, HTML extraction and
chunk overlap, FTS expression building (including malformed input that must not
break the MATCH parser), RRF, LSA matching a query with no shared keywords,
artifact cache round-trip, duplicate collapsing, cursor round-trip and garbage
rejection, score ties, a full forward walk that must cover every row exactly once
with no repeats, cross-query and stale cursor handling, and that each cache layer
hits when it should.

The UI was verified in a browser end to end: page 1 to page 2 shows ranks 11-20
with zero URL overlap, Previous round-trips to exactly the page 1 result set and
disables itself there, the empty state renders for a no-match query, and the
status bar reports which cache layers served the request.

## Known limits

- **Client-rendered pages.** The site is Webflow-rendered; some listing pages
  expose few static anchors, and a number of linked paths 404 for a non-JS
  client. Link-based BFS still finds >1300 in-scope URLs, so the 50-page budget
  is the binding constraint, not discovery. `sitemap.xml` and the site's
  `llms.txt` are both advertised in `robots.txt` and would be the natural
  frontier backstop if discovery ever starved.
- **LSA is corpus-bound.** It can only relate terms that co-occur in the 50
  crawled pages. A real deployment would swap in sentence embeddings behind the
  same interface.
- **Single process, single writer.** SQLite + an in-process index is right for
  50 pages and wrong for 50,000; at that size the index moves out and the
  snapshot cache becomes Redis.
- **Rerank weights are hand-set**, not fit to relevance judgments.
- `max_depth` is 4 and the frontier is re-derived per crawl rather than
  persisted, so a resumed crawl re-walks known pages (cheaply, via layer 2).
