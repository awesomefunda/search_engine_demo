# Deploying the interview demo

## Vercel

The root `app.py` exports the existing FastAPI application. This fixes
`PYTHON_ENTRYPOINT_NOT_FOUND` without changing the UI or API routes.

1. Copy `app.py`, `vercel.json`, and the updated `fivetran_search/config.py`
   into the repository root and matching package directory, then commit/push.
2. In Vercel, set Root Directory to the folder containing `app.py` and
   `requirements.txt`. Use the FastAPI preset; leave build and output
   overrides unset. Redeploy the new commit.
3. Open `/`, then `/api/status`, then search for `pricing`.
4. Search for a broad term such as `data`, follow Next, and verify the result
   pages do not overlap. Repeat the query and inspect the cache indicator.

On Vercel, SQLite and LSA artifacts use the temporary directory by default.
Do not point `FTS_DATA_DIR` at the deployed source directory. The 300-second
function setting allows a longer crawl; actual execution remains subject to
the deployment's platform limits. Failed/slow fetches can still exceed it.

## Demo limitation to understand before sharing

`data/` is intentionally excluded from Git. A new function instance therefore
starts empty and performs the crawl on its first query. The crawler is
synchronous, so that query can be slow. Temporary SQLite and query snapshots
are not shared between instances and disappear when an instance is recycled.
Consequently Vercel does not guarantee cross-instance caching or stable cursor
pagination for this architecture. This patch fixes startup and filesystem
compatibility, not those persistence limitations.

For a dependable manager demo with the current SQLite design, deploy the same
app as a single long-running Python instance with a persistent writable disk:

```sh
pip install -r requirements.txt
export FTS_DATA_DIR=/your-mounted-disk/fivetran-search
python -m fivetran_search.cli crawl
uvicorn app:app --host 0.0.0.0 --port "$PORT" --workers 1
```

Run the crawl once before sharing the link. For a scalable Vercel deployment,
move the corpus and query snapshots into shared durable storage and run crawl
refreshes through a durable job system. Local temporary files remain only an
optimization.

The public `/api/refresh` endpoint can trigger an expensive crawl; protect or
disable it before making a widely accessible deployment.
