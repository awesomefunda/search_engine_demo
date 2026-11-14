# Fivetran Site Search

An MVP search engine for `fivetran.com`. Enter a keyword or a natural-language
query to search a bounded website corpus, view relevance-ranked results, and
navigate with cursor-based pagination at 10 results per page.

[Hosted demo](https://search-engine-demo.vercel.app/) ·
[Technical documentation](fivetran-search/README.md) ·
[Deployment guide](fivetran-search/DEPLOYMENT.md)

> **Try the demo:** Search for `data` or `how much does it cost`, then use
> Next/Previous to navigate. A live 50-page crawl, keyword and sentence search,
> ranking explanations, and cached pagination have been verified. A cold
> instance may take about 30 seconds for its initial crawl.

## What the MVP implements

- Recursive link discovery from `https://www.fivetran.com/`, indexing up to
  50 HTML pages within a bounded crawl.
- Hybrid retrieval: BM25 keyword search and TF-IDF/LSA semantic similarity.
- Reciprocal rank fusion followed by feature-based reranking.
- Result snippets, ranking explanations, and identical-content deduplication.
- Cursor-based forward and backward navigation, 10 results per page by default.
- Layered caching for page freshness, conditional HTTP requests, index artifacts,
  and ranked query snapshots.
- A FastAPI backend, browser UI, CLI, and 33 offline automated tests.

No external model API, API key, or model download is required.

## Run locally

Use Python 3.14, matching the hosted deployment runtime.

```sh
git clone https://github.com/awesomefunda/search_engine_demo.git
cd search_engine_demo/fivetran-search
python -m pip install -r requirements.txt
python -m fivetran_search.cli crawl
uvicorn app:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. Check the crawl output before searching: indexing
requires that the site permits the crawler and that pages are reachable. If the
corpus is absent or stale, a search attempts to crawl automatically.

Example queries: `salesforce connector`, `how much does it cost`, and
`data pipeline security and compliance`. Results depend on the pages indexed.

Run the search tests from `fivetran-search/`:

```sh
python -m unittest discover -s tests -v
```

## Design at a glance

Crawl and extract pages → store documents and chunks → retrieve keyword and
semantic candidates → fuse and rerank → cache the ranked list → return a page.

SQLite with FTS5 keeps documents, keyword indexing, and snapshots in one
embedded database, which fits a 50-page MVP without a separate service. LSA
provides a lightweight corpus-based similarity signal; it does not offer the
language understanding of a pretrained embedding model. The reranker exposes
its features so relevance decisions can be inspected.

The current persistence model is suited to a single server with writable disk.
Vercel uses instance-local temporary storage, so cache persistence and cursor
continuity are not guaranteed across instances or restarts. See the deployment
guide for details and the path to shared durable storage.

## Repository contents

The search application, tests, and screenshots are under [`fivetran-search/`](fivetran-search/).
Earlier learning materials remain under `lessons/`, `exercises/`, `solutions/`,
and `tests/`; their original guide is preserved in [AUTOGRAD.md](AUTOGRAD.md).

This is an independent demonstration project and is not affiliated with Fivetran.
