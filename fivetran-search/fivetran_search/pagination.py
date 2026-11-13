"""Cursor-based pagination over a ranked snapshot.

Offset pagination on a live ranking drifts: if the index changes between page 1
and page 2, rows shift and the user sees duplicates or gaps. Instead:

  1. the full ranked list for a query is computed once and snapshotted
     (keyed by query + corpus_version),
  2. the cursor carries the sort key of the last row the client saw,
  3. the next page is a keyset scan for rows strictly after that key.

The cursor also carries the corpus_version it was minted against. If the corpus
has been recrawled since, the cursor is reported as stale instead of silently
pointing into a different ranking.
"""
from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass

CURSOR_VERSION = 1


class InvalidCursor(ValueError):
    pass


@dataclass(frozen=True)
class Cursor:
    query_key: str          # hash of the normalized query
    corpus_version: str
    last_score: float
    last_page_id: int
    offset: int             # informational: how many rows precede this page

    def encode(self) -> str:
        payload = {
            "v": CURSOR_VERSION,
            "q": self.query_key,
            "cv": self.corpus_version,
            "s": round(self.last_score, 9),
            "d": self.last_page_id,
            "o": self.offset,
        }
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    @staticmethod
    def decode(token: str) -> "Cursor":
        try:
            pad = "=" * (-len(token) % 4)
            payload = json.loads(base64.urlsafe_b64decode(token + pad))
        except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
            raise InvalidCursor("cursor is not decodable") from exc
        if payload.get("v") != CURSOR_VERSION:
            raise InvalidCursor("unsupported cursor version")
        try:
            return Cursor(
                query_key=str(payload["q"]),
                corpus_version=str(payload["cv"]),
                last_score=float(payload["s"]),
                last_page_id=int(payload["d"]),
                offset=int(payload.get("o", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidCursor("cursor is missing fields") from exc


def sort_key(row: dict) -> tuple[float, int]:
    """Ranking order: score descending, page_id ascending as the tie-break.

    A total order is what makes the keyset scan correct; ties on score alone
    would let two rows be mutually 'after' each other.
    """
    return (-float(row["score"]), int(row["page_id"]))


def page_after(rows: list[dict], cursor: Cursor | None, page_size: int) -> tuple[list[dict], int]:
    """Returns (page_rows, offset_of_first_row). `rows` must already be sorted."""
    if cursor is None:
        return rows[:page_size], 0
    target = (-cursor.last_score, cursor.last_page_id)
    start = len(rows)
    for i, row in enumerate(rows):
        if sort_key(row) > target:
            start = i
            break
    return rows[start:start + page_size], start
