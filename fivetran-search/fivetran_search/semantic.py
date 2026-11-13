"""Semantic retrieval: TF-IDF over chunks, reduced with LSA (truncated SVD).

Why LSA and not a transformer: this has to run with no API key, no model
download and no GPU, and still match a sentence like "how much does it cost"
to a page that says "pricing" and "MAR consumption". Projecting TF-IDF into a
low-rank latent space does that, because terms that co-occur across the corpus
end up with similar directions.

The embedder is a Strategy: EmbeddingIndex is the interface the pipeline uses,
so a sentence-transformers backend can be dropped in later without the search
path changing.

Artifacts are cached to an .npz keyed by corpus_version, so an unchanged
corpus never pays for a rebuild.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import SETTINGS
from .textutil import tokenize


# --------------------------------------------------------------------------
# sparse helpers (CSR built by hand: scipy is not a dependency here)
# --------------------------------------------------------------------------
@dataclass
class Csr:
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray
    shape: tuple[int, int]

    def matmul(self, b: np.ndarray) -> np.ndarray:
        """X @ B for dense B of shape (V, k)."""
        n = self.shape[0]
        out = np.zeros((n, b.shape[1]), dtype=np.float32)
        for i in range(n):
            s, e = self.indptr[i], self.indptr[i + 1]
            if e > s:
                out[i] = self.data[s:e] @ b[self.indices[s:e]]
        return out

    def rmatmul(self, a: np.ndarray) -> np.ndarray:
        """X.T @ A for dense A of shape (n, k). Row indices are unique per row,
        so the scatter-add can use plain += without np.add.at."""
        out = np.zeros((self.shape[1], a.shape[1]), dtype=np.float32)
        for i in range(self.shape[0]):
            s, e = self.indptr[i], self.indptr[i + 1]
            if e > s:
                out[self.indices[s:e]] += np.outer(self.data[s:e], a[i])
        return out


def randomized_svd(x: Csr, k: int, n_oversample: int = 10,
                   n_iter: int = 2, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Halko-style randomized truncated SVD. Returns (U, S, Vt) with k columns.

    Cost is O(nnz * k) instead of the O(n^2 V) a full SVD would need, which is
    what makes a rebuild cheap enough to do inline on a cache miss.
    """
    rng = np.random.default_rng(seed)
    n, v = x.shape
    k = max(1, min(k, min(n, v) - 1)) if min(n, v) > 1 else 1
    p = min(k + n_oversample, min(n, v))

    omega = rng.standard_normal((v, p)).astype(np.float32)
    q, _ = np.linalg.qr(x.matmul(omega))
    for _ in range(n_iter):
        z, _ = np.linalg.qr(x.rmatmul(q))
        q, _ = np.linalg.qr(x.matmul(z))

    b = x.rmatmul(q).T                      # (p, V)
    ub, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ ub
    return u[:, :k], s[:k], vt[:k]


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------
class EmbeddingIndex:
    """TF-IDF + LSA index over chunks, with page-level roll-up."""

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0, dtype=np.float32)
        self.components: np.ndarray = np.zeros((0, 0), dtype=np.float32)   # (k, V)
        self.chunk_emb: np.ndarray = np.zeros((0, 0), dtype=np.float32)    # (n, k) normalized
        self.chunk_page_ids: np.ndarray = np.zeros(0, dtype=np.int64)
        self.corpus_version: str = ""
        self.n_chunks: int = 0

    # ---- build -----------------------------------------------------------
    def build(self, chunks: list[tuple[int, int, str]], corpus_version: str) -> "EmbeddingIndex":
        self.corpus_version = corpus_version
        if not chunks:
            return self

        tokenized = [tokenize(text) for _, _, text in chunks]
        self.chunk_page_ids = np.array([pid for _, pid, _ in chunks], dtype=np.int64)
        n = len(tokenized)
        self.n_chunks = n

        df: dict[str, int] = {}
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1

        min_df = SETTINGS.min_df if n > 2 * SETTINGS.min_df else 1
        candidates = [(t, d) for t, d in df.items() if d >= min_df]
        # Cap vocabulary by document frequency; idf handles the rest.
        candidates.sort(key=lambda kv: (-kv[1], kv[0]))
        candidates = candidates[: SETTINGS.max_vocab]
        self.vocab = {t: i for i, t in enumerate(sorted(t for t, _ in candidates))}
        v = len(self.vocab)
        if v == 0:
            return self

        self.idf = np.zeros(v, dtype=np.float32)
        for t, i in self.vocab.items():
            self.idf[i] = math.log((1.0 + n) / (1.0 + df[t])) + 1.0

        # Build CSR of L2-normalized TF-IDF rows.
        indptr = np.zeros(n + 1, dtype=np.int64)
        idx_list: list[np.ndarray] = []
        dat_list: list[np.ndarray] = []
        for i, toks in enumerate(tokenized):
            counts: dict[int, int] = {}
            for t in toks:
                j = self.vocab.get(t)
                if j is not None:
                    counts[j] = counts.get(j, 0) + 1
            if counts:
                cols = np.fromiter(counts.keys(), dtype=np.int64, count=len(counts))
                order = np.argsort(cols)
                cols = cols[order]
                tf = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))[order]
                vals = (1.0 + np.log(tf)) * self.idf[cols]
                norm = float(np.linalg.norm(vals)) or 1.0
                vals = (vals / norm).astype(np.float32)
            else:
                cols = np.zeros(0, dtype=np.int64)
                vals = np.zeros(0, dtype=np.float32)
            idx_list.append(cols)
            dat_list.append(vals)
            indptr[i + 1] = indptr[i] + len(cols)

        x = Csr(indptr, np.concatenate(idx_list) if idx_list else np.zeros(0, dtype=np.int64),
                np.concatenate(dat_list) if dat_list else np.zeros(0, dtype=np.float32), (n, v))

        k = min(SETTINGS.lsa_components, max(2, min(n, v) - 1))
        u, s, vt = randomized_svd(x, k)
        self.components = np.ascontiguousarray(vt.astype(np.float32))      # (k, V)
        self.chunk_emb = l2_normalize((u * s).astype(np.float32))          # (n, k)
        return self

    # ---- query -----------------------------------------------------------
    def _vectorize(self, text: str) -> np.ndarray:
        v = len(self.vocab)
        vec = np.zeros(v, dtype=np.float32)
        if v == 0:
            return vec
        counts: dict[int, int] = {}
        for t in tokenize(text):
            j = self.vocab.get(t)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
        for j, c in counts.items():
            vec[j] = (1.0 + math.log(c)) * self.idf[j]
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    def embed_query(self, text: str) -> np.ndarray:
        """Project the query TF-IDF vector into the latent space."""
        if self.components.size == 0:
            return np.zeros(0, dtype=np.float32)
        q = self._vectorize(text) @ self.components.T          # (k,)
        norm = float(np.linalg.norm(q))
        return q / norm if norm else q

    def search_chunks(self, text: str, limit: int) -> list[tuple[int, int, float]]:
        """Returns [(chunk_row, page_id, cosine)] best-first."""
        if self.chunk_emb.size == 0:
            return []
        q = self.embed_query(text)
        if q.size == 0 or not np.any(q):
            return []
        sims = self.chunk_emb @ q
        limit = min(limit, sims.shape[0])
        top = np.argpartition(-sims, limit - 1)[:limit]
        top = top[np.argsort(-sims[top])]
        return [(int(i), int(self.chunk_page_ids[i]), float(sims[i])) for i in top]

    def search_pages(self, text: str, limit: int) -> list[tuple[int, float]]:
        """Chunk hits rolled up to pages by max-pooling (best passage wins)."""
        hits = self.search_chunks(text, max(limit * 4, 80))
        best: dict[int, float] = {}
        for _row, pid, score in hits:
            if score > best.get(pid, -1e9):
                best[pid] = score
        ranked = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:limit]

    def best_chunk_score(self, page_id: int, query_vec: np.ndarray) -> float:
        if self.chunk_emb.size == 0 or query_vec.size == 0:
            return 0.0
        mask = self.chunk_page_ids == page_id
        if not np.any(mask):
            return 0.0
        return float(np.max(self.chunk_emb[mask] @ query_vec))

    # ---- artifact cache --------------------------------------------------
    def path_for(self, cache_dir: Path) -> Path:
        return cache_dir / f"lsa-{self.corpus_version}.npz"

    def save(self, cache_dir: Path) -> Path:
        cache_dir.mkdir(parents=True, exist_ok=True)
        p = self.path_for(cache_dir)
        terms = np.array(sorted(self.vocab, key=self.vocab.get), dtype=object)
        np.savez_compressed(
            p, terms=terms, idf=self.idf, components=self.components,
            chunk_emb=self.chunk_emb, chunk_page_ids=self.chunk_page_ids,
            corpus_version=np.array(self.corpus_version),
        )
        return p

    @classmethod
    def load(cls, cache_dir: Path, corpus_version: str) -> "EmbeddingIndex | None":
        p = cache_dir / f"lsa-{corpus_version}.npz"
        if not p.exists():
            return None
        try:
            z = np.load(p, allow_pickle=True)
            idx = cls()
            idx.vocab = {t: i for i, t in enumerate(z["terms"].tolist())}
            idx.idf = z["idf"]
            idx.components = z["components"]
            idx.chunk_emb = z["chunk_emb"]
            idx.chunk_page_ids = z["chunk_page_ids"]
            idx.corpus_version = str(z["corpus_version"])
            idx.n_chunks = int(idx.chunk_emb.shape[0])
            return idx
        except Exception:
            return None
