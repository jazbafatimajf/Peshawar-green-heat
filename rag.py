"""Tiny retrieval layer over knowledge/*.md (TF-IDF, no external services).

Each chunk is one '## heading' section and gets a stable id like 'species_guidance#2'.
The agent may only cite ids that retrieval actually returned (see agent.py guardrails).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import ROOT

KB_DIR = ROOT / "knowledge"


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    text: str
    source: str


def _load_chunks(kb_dir: Path) -> List[Chunk]:
    chunks: List[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        raw = path.read_text(encoding="utf-8")
        m = re.search(r"^source:\s*(.+)$", raw, flags=re.M)
        source = m.group(1).strip() if m else path.stem
        parts = re.split(r"^## ", raw, flags=re.M)[1:]
        for n, part in enumerate(parts, start=1):
            title, _, body = part.partition("\n")
            chunks.append(Chunk(f"{path.stem}#{n}", title.strip(), body.strip(), source))
    return chunks


class KnowledgeBase:
    def __init__(self, kb_dir: Path = KB_DIR):
        self.chunks = _load_chunks(kb_dir)
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._mat = self._vec.fit_transform([f"{c.title}. {c.text}" for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> List[dict]:
        if not self.chunks:
            return []
        sims = cosine_similarity(self._vec.transform([query]), self._mat)[0]
        order = sims.argsort()[::-1][:k]
        return [
            {"id": self.chunks[i].id, "title": self.chunks[i].title, "text": self.chunks[i].text,
             "source": self.chunks[i].source, "score": round(float(sims[i]), 3)}
            for i in order if sims[i] >= min_score
        ]


@lru_cache(maxsize=1)
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()
