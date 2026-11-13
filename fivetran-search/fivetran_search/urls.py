"""URL normalization and scope rules.

Normalization is what stops a 50-page budget being burnt on the same page
reached five different ways (trailing slash, utm tags, #anchor, http vs https).
"""
from __future__ import annotations

from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

from .config import DROP_QUERY_PARAM_PREFIXES, SETTINGS, path_is_denied


def _canonical_host(host: str) -> str | None:
    """Map any accepted spelling of the site's host to the one canonical form.

    Returns None for offsite hosts.
    """
    bare = host[4:] if host.startswith("www.") else host
    allowed = {h[4:] if h.startswith("www.") else h for h in SETTINGS.allowed_hosts}
    return SETTINGS.canonical_host if bare in allowed else None


def normalize(url: str, base: str | None = None) -> str | None:
    """Canonical form of a URL, or None if it is not a crawlable web page."""
    if not url:
        return None
    url = url.strip()
    if url.startswith(("mailto:", "tel:", "javascript:", "data:", "#")):
        return None
    if base:
        url = urljoin(base, url)

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https"):
        return None

    host = (parts.hostname or "").lower()
    if not host:
        return None
    # Offsite hosts are kept verbatim; in_scope() is what rejects them. Known
    # hosts are rewritten to the canonical spelling so apex and www never both
    # consume crawl budget.
    host = _canonical_host(host) or host

    path = parts.path or "/"
    # Collapse duplicate slashes, drop a trailing slash except on root.
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/") or "/"

    # Keep meaningful query params, drop tracking noise, sort for stability.
    kept = [
        (k, v) for k, v in parse_qsl(parts.query, keep_blank_values=False)
        if not k.lower().startswith(DROP_QUERY_PARAM_PREFIXES)
    ]
    query = urlencode(sorted(kept))

    # Fragments never address a different document.
    return urlunsplit(("https", host, path, query, ""))


def in_scope(url: str) -> bool:
    """Same-site and not matched by a deny pattern."""
    parts = urlsplit(url)
    if _canonical_host((parts.hostname or "").lower()) is None:
        return False
    return not path_is_denied(parts.path or "/")


def path_of(url: str) -> str:
    return urlsplit(url).path or "/"


def depth_of(url: str) -> int:
    return len([p for p in path_of(url).split("/") if p])
