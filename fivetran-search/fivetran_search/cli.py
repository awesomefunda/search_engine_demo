"""Command line front end.

    python -m fivetran_search.cli crawl  [--force] [--max-pages N]
    python -m fivetran_search.cli search "how much does it cost" [--page 2] [--json]
    python -m fivetran_search.cli status
"""
from __future__ import annotations

import argparse
import json
import sys

from .pipeline import SearchService

# Windows consoles default to cp1252, which cannot encode the ellipsis and
# arrows that appear in page text. Never let printing crash a search.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _print_results(resp, show_features: bool) -> None:
    c = resp.cache
    print(f'"{resp.query}"  ->  {resp.total_results} results '
          f'(showing {resp.offset + 1}-{resp.offset + len(resp.results)})')
    print(f"  cache: crawl_skipped={c.get('crawl_skipped')} index={c.get('index')} "
          f"snapshot={c.get('query_snapshot')}  |  {resp.timings_ms.get('total')} ms")
    if resp.warnings:
        for w in resp.warnings:
            print(f"  ! {w}")
    print()
    for i, r in enumerate(resp.results, start=resp.offset + 1):
        print(f"{i:>3}. {r['title']}")
        print(f"     {r['url']}")
        print(f"     score {r['score']:.4f}  [{', '.join(r['sources'])}]")
        if r["snippet"]:
            print(f"     {r['snippet']}")
        if show_features:
            print(f"     features: {r['features']}")
        print()
    if resp.next_cursor:
        print(f"next cursor: {resp.next_cursor}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fivetran-search")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_crawl = sub.add_parser("crawl", help="build or refresh the corpus")
    p_crawl.add_argument("--force", action="store_true",
                         help="refetch every page, ignoring per-page TTL")
    p_crawl.add_argument("--max-pages", type=int, default=None)

    p_search = sub.add_parser("search", help="run a query")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("--page", type=int, default=1,
                          help="walks the cursor chain N-1 times")
    p_search.add_argument("--page-size", type=int, default=None)
    p_search.add_argument("--features", action="store_true", help="show the score breakdown")
    p_search.add_argument("--json", action="store_true")

    sub.add_parser("status", help="corpus and cache state")

    args = parser.parse_args(argv)
    service = SearchService()

    if args.cmd == "crawl":
        report = service.crawler.crawl(force=args.force, max_pages=args.max_pages)
        service.store.purge_query_cache()
        service.ensure_index(report.corpus_version)
        print(json.dumps(report.as_dict(), indent=2))
        return 0

    if args.cmd == "status":
        print(json.dumps(service.status(), indent=2, default=str))
        return 0

    query = " ".join(args.query)
    resp = service.search(query, page_size=args.page_size)
    for _ in range(max(0, args.page - 1)):
        if not resp.next_cursor:
            break
        resp = service.search(query, cursor_token=resp.next_cursor, page_size=args.page_size)

    if args.json:
        print(json.dumps(resp.as_dict(), indent=2, default=str))
    else:
        _print_results(resp, args.features)
    return 0


if __name__ == "__main__":
    sys.exit(main())
