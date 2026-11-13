"""Polite HTTP fetching with the first cache layer: conditional GET.

A page already in the store is revalidated with If-None-Match / If-Modified-Since.
A 304 costs one small round trip and no parsing, which is what makes a recrawl
cheap enough to run on a schedule.
"""
from __future__ import annotations

import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests

from .config import SETTINGS


@dataclass
class FetchResult:
    url: str
    status: int
    body: bytes | None          # None on 304 / error
    etag: str | None
    last_modified: str | None
    content_type: str
    from_cache: bool            # True when the server answered 304
    error: str | None = None

    @property
    def is_html(self) -> bool:
        return "html" in (self.content_type or "").lower()


class RobotsPolicy:
    """robots.txt per host, parsed once and reused."""

    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self._parsers: dict[str, robotparser.RobotFileParser | None] = {}
        self._delays: dict[str, float] = {}

    def _parser(self, url: str) -> robotparser.RobotFileParser | None:
        host = urlsplit(url).netloc
        if host in self._parsers:
            return self._parsers[host]
        rp = robotparser.RobotFileParser()
        robots_url = f"{urlsplit(url).scheme}://{host}/robots.txt"
        try:
            resp = requests.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                timeout=SETTINGS.request_timeout,
            )
            if resp.status_code >= 400:
                rp = None  # no robots.txt -> allow
            else:
                rp.parse(resp.text.splitlines())
        except requests.RequestException:
            rp = None
        self._parsers[host] = rp
        if rp is not None:
            delay = rp.crawl_delay(self.user_agent)
            if delay:
                self._delays[host] = float(delay)
        return rp

    def allowed(self, url: str) -> bool:
        rp = self._parser(url)
        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)

    def delay_for(self, url: str) -> float:
        host = urlsplit(url).netloc
        return max(SETTINGS.crawl_delay, self._delays.get(host, 0.0))


class Fetcher:
    def __init__(self, robots: RobotsPolicy | None = None):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = SETTINGS.user_agent
        self.session.headers["Accept"] = "text/html,application/xhtml+xml"
        self.robots = robots or RobotsPolicy(SETTINGS.user_agent)
        self._last_request_at: dict[str, float] = {}
        self.stats = {"fetched": 0, "revalidated_304": 0, "errors": 0,
                      "robots_blocked": 0, "retries": 0}

    def _wait_turn(self, url: str) -> None:
        host = urlsplit(url).netloc
        delay = self.robots.delay_for(url)
        last = self._last_request_at.get(host)
        if last is not None:
            sleep_for = delay - (time.monotonic() - last)
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._last_request_at[host] = time.monotonic()

    def fetch(self, url: str, etag: str | None = None,
              last_modified: str | None = None) -> FetchResult:
        if not self.robots.allowed(url):
            self.stats["robots_blocked"] += 1
            return FetchResult(url, 0, None, None, None, "", False, error="blocked by robots.txt")

        headers: dict[str, str] = {}
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

        # Transient TLS/connection resets are common enough on a 50-page walk
        # that one retry with backoff is worth more than a lost page.
        resp = None
        last_exc: Exception | None = None
        for attempt in range(SETTINGS.fetch_attempts):
            self._wait_turn(url)
            try:
                resp = self.session.get(
                    url, headers=headers, timeout=SETTINGS.request_timeout,
                    allow_redirects=True, stream=True,
                )
                break
            except requests.RequestException as exc:
                last_exc = exc
                self.stats["retries"] += 1
                time.sleep(SETTINGS.retry_backoff * (attempt + 1))
        if resp is None:
            self.stats["errors"] += 1
            self.stats["retries"] -= 1  # the final failure is counted as an error
            return FetchResult(url, 0, None, None, None, "", False,
                               error=type(last_exc).__name__ if last_exc else "unknown")

        ctype = resp.headers.get("Content-Type", "")
        new_etag = resp.headers.get("ETag")
        new_lm = resp.headers.get("Last-Modified")

        if resp.status_code == 304:
            resp.close()
            self.stats["revalidated_304"] += 1
            return FetchResult(url, 304, None, etag, last_modified, ctype, True)

        if resp.status_code >= 400:
            resp.close()
            self.stats["errors"] += 1
            return FetchResult(url, resp.status_code, None, new_etag, new_lm, ctype, False,
                               error=f"HTTP {resp.status_code}")

        if "html" not in ctype.lower():
            resp.close()
            return FetchResult(url, resp.status_code, None, new_etag, new_lm, ctype, False,
                               error="not html")

        body = b""
        for chunk in resp.iter_content(65536):
            body += chunk
            if len(body) > SETTINGS.max_bytes_per_page:
                break
        resp.close()
        self.stats["fetched"] += 1
        return FetchResult(url, resp.status_code, body, new_etag, new_lm, ctype, False)
