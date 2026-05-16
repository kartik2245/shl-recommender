"""
Catalog loader + hybrid retrieval.

Two retrieval modes:
  - retrieve(query, k): semantic + lexical hybrid for "recommend me X"
  - find_by_name(name): fuzzy name lookup for "compare OPQ vs GSA"

Why hybrid? The catalog mixes precise product names ("Java 8 (New)",
"OPQ32r") with conceptual descriptions ("works with stakeholders").
BM25 nails the names; dense embeddings nail the intent. Reciprocal-rank
fusion blends them without tuning weights.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rank_bm25 import BM25Okapi


# Lazy import -- sentence-transformers is heavy and we only need it on
# server startup, not at import time (helps tests stay fast).
def _load_encoder():
    from sentence_transformers import SentenceTransformer
    # all-MiniLM-L6-v2: 80MB, ~14k sentences/sec on CPU. Good default.
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@dataclass
class Item:
    name: str
    url: str
    test_types: list[str]
    description: str
    job_levels: list[str]
    languages: list[str]
    assessment_length: str
    remote_testing: bool
    adaptive_irt: bool

    def as_recommendation_dict(self) -> dict:
        """Shape required by the SHL response schema."""
        # The spec shows test_type as a single letter. If an item has
        # multiple codes we pick the first (most representative). The
        # full list stays accessible via Item.test_types for the agent's
        # natural-language reply.
        primary = self.test_types[0] if self.test_types else ""
        return {"name": self.name, "url": self.url, "test_type": primary}

    def to_search_text(self) -> str:
        """Concatenate fields we want to be searchable."""
        parts = [
            self.name,
            self.description,
            " ".join(self.job_levels),
            " ".join(self.test_types),
            self.assessment_length,
        ]
        return " | ".join(p for p in parts if p)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Catalog:
    """
    In-memory catalog with two indices.

    Build once at server startup; query per request. Thread-safe for reads
    because numpy arrays + python lists are read-only after __init__.
    """

    def __init__(self, items: list[Item], legend: dict[str, str]):
        if not items:
            raise ValueError("Catalog has zero items. Run scripts/scrape_catalog.py first.")
        self.items = items
        self.legend = legend

        # Lexical index -- BM25 over tokenized search text.
        self._tokenized = [_tokenize(it.to_search_text()) for it in items]
        self._bm25 = BM25Okapi(self._tokenized)

        # Dense index -- L2-normalized embeddings -> cosine via dot product.
        encoder = _load_encoder()
        texts = [it.to_search_text() for it in items]
        emb = encoder.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        self._embeddings: np.ndarray = np.asarray(emb, dtype=np.float32)
        self._encoder = encoder

        # Name index for compare/lookup.
        self._name_lower = [it.name.lower() for it in items]

    # ---------- construction ----------

    @classmethod
    def from_json(cls, path: str | Path) -> "Catalog":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        items = [
            Item(
                name=raw["name"],
                url=raw["url"],
                test_types=list(raw.get("test_types", [])),
                description=raw.get("description", ""),
                job_levels=list(raw.get("job_levels", [])),
                languages=list(raw.get("languages", [])),
                assessment_length=raw.get("assessment_length", ""),
                remote_testing=bool(raw.get("remote_testing", False)),
                adaptive_irt=bool(raw.get("adaptive_irt", False)),
            )
            for raw in data["items"]
        ]
        return cls(items, data.get("test_type_legend", {}))

    # ---------- retrieval ----------

    def retrieve(self, query: str, k: int = 10) -> list[tuple[Item, float]]:
        """
        Hybrid retrieval via reciprocal rank fusion (RRF).
        Returns (item, fused_score) sorted desc, top-k.
        """
        if not query.strip():
            return []

        # 1. BM25 ranking
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        bm25_order = np.argsort(-bm25_scores)

        # 2. Dense ranking
        q_emb = self._encoder.encode([query], normalize_embeddings=True)[0]
        dense_scores = self._embeddings @ q_emb  # cosine sim (both L2-normalized)
        dense_order = np.argsort(-dense_scores)

        # 3. RRF fuse: score(i) = sum over rankers of 1 / (60 + rank_i)
        # 60 is the standard RRF constant from Cormack et al. 2009.
        K_RRF = 60
        fused = np.zeros(len(self.items), dtype=np.float32)
        for rank, idx in enumerate(bm25_order):
            fused[idx] += 1.0 / (K_RRF + rank)
        for rank, idx in enumerate(dense_order):
            fused[idx] += 1.0 / (K_RRF + rank)

        # 4. Top-k
        top_idx = np.argsort(-fused)[:k]
        return [(self.items[i], float(fused[i])) for i in top_idx]

    def filter_retrieve(
        self,
        query: str,
        *,
        k: int = 10,
        required_types: Iterable[str] | None = None,
        max_minutes: int | None = None,
    ) -> list[tuple[Item, float]]:
        """
        Retrieve then filter. Used for refinement turns like 'add personality'
        or 'under 30 minutes'.
        """
        pool = self.retrieve(query, k=max(k * 3, 30))  # over-fetch for filtering
        out = []
        required = set(t.upper() for t in (required_types or []))
        for item, score in pool:
            if required and not required.issubset(set(item.test_types)):
                continue
            if max_minutes is not None:
                mins = _parse_minutes(item.assessment_length)
                if mins is not None and mins > max_minutes:
                    continue
            out.append((item, score))
            if len(out) >= k:
                break
        return out

    def find_by_name(self, query_name: str, k: int = 5) -> list[Item]:
        """Fuzzy name lookup for compare flows."""
        q = query_name.lower().strip()
        if not q:
            return []
        # Substring first
        exact = [it for it, lo in zip(self.items, self._name_lower) if q in lo]
        if exact:
            return exact[:k]
        # Fall back to retrieval over names
        return [it for it, _ in self.retrieve(query_name, k=k)]


def _parse_minutes(text: str) -> int | None:
    """Extract a minute count from strings like 'Approximate Completion Time in minutes = 35'."""
    if not text:
        return None
    m = re.search(r"(\d+)\s*(?:minutes|mins?|min)\b", text, flags=re.I)
    if m:
        return int(m.group(1))
    m = re.search(r"=\s*(\d+)", text)
    if m:
        return int(m.group(1))
    return None
