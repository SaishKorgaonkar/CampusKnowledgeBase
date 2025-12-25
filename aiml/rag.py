import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import faiss

from embedder import Embedder


def _load_jsonl(path: str) -> List[Dict]:
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_num}: {e}") from e
    return records


def _distance_to_relevance(distance: float) -> float:
    # FAISS IndexFlatL2 returns squared L2 distance. Map to (0,1] with a simple monotonic transform.
    # 0 -> 1.0, larger distance -> closer to 0.
    if distance < 0:
        distance = 0.0
    return 1.0 / (1.0 + float(distance))


def _repo_root() -> Path:
    # aiml/rag.py -> aiml/ -> repo root
    return Path(__file__).resolve().parent.parent


def _candidate_paths(data_dir: Path, course: str, semester: str) -> Tuple[Path, Path]:
    # Preferred: data/{course}/{semester}/{course}_{semester}_*.{index|jsonl}
    # Provide a fallback variant where spaces in semester are replaced with '-' for filenames.
    sem_for_filename = semester.replace(" ", "-")

    folder = data_dir / course / semester

    chunks_candidates = [
        folder / f"{course}_{semester}_chunks.jsonl",
        folder / f"{course}_{sem_for_filename}_chunks.jsonl",
    ]

    index_candidates = [
        folder / f"{course}_{semester}_faiss.index",
        folder / f"{course}_{sem_for_filename}_faiss.index",
    ]

    chunks_path = next((p for p in chunks_candidates if p.exists()), chunks_candidates[0])
    index_path = next((p for p in index_candidates if p.exists()), index_candidates[0])
    return index_path, chunks_path


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index_path: str = "ingestion/output/faiss.index",
        chunks_path: str = "ingestion/output/chunks.jsonl",
    ):
        self.embedder = embedder
        # Defaults/fallback when course+semester aren't provided
        self.default_index_path = index_path
        self.default_chunks_path = chunks_path

        # Cache loaded resources by (course, semester) or by resolved fallback paths
        self._cache: dict[tuple, tuple] = {}

    def _load_resources(self, course: Optional[str], semester: Optional[str]):
        if course and semester:
            data_dir = _repo_root() / "data"
            index_path, chunks_path = _candidate_paths(data_dir, course, semester)
            cache_key = ("data", str(index_path), str(chunks_path))
        else:
            index_path = _repo_root() / "aiml" / self.default_index_path
            chunks_path = _repo_root() / "aiml" / self.default_chunks_path
            cache_key = ("fallback", str(index_path), str(chunks_path))

        if cache_key in self._cache:
            return self._cache[cache_key]

        index = None
        chunks: List[Dict] = []

        try:
            if not index_path.exists():
                raise FileNotFoundError(str(index_path))
            index = faiss.read_index(str(index_path))
        except Exception as e:
            print(f"FAISS index not found at {index_path}: {e}")

        try:
            if not chunks_path.exists():
                raise FileNotFoundError(str(chunks_path))
            if str(chunks_path).lower().endswith(".jsonl"):
                chunks = _load_jsonl(str(chunks_path))
            else:
                with open(chunks_path, "r", encoding="utf-8") as f:
                    chunks = json.load(f)
        except Exception as e:
            print(f"Chunks file not found at {chunks_path}: {e}")

        self._cache[cache_key] = (index, chunks)
        return index, chunks

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        course: Optional[str] = None,
        semester: Optional[str] = None,
    ) -> List[Dict]:
        index, chunks = self._load_resources(course, semester)

        if index is None:
            print("Cannot retrieve: FAISS index not loaded.")
            return []
        if not chunks:
            print("Cannot retrieve: chunks metadata not loaded.")
            return []

        q_vec = self.embedder.embed_text(query).reshape(1, -1)
        distances, indices = index.search(q_vec, top_k)

        results: List[Dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            results.append({**chunk, "relevance": _distance_to_relevance(float(dist))})

        return results
