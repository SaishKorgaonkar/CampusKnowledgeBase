import os
import json
import faiss
from typing import List, Dict
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


class Retriever:
    def __init__(
        self,
        embedder: Embedder,
        index_path: str = "ingestion/output/faiss.index",
        chunks_path: str = "ingestion/output/chunks.jsonl",
    ):
        self.embedder = embedder
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.chunks: List[Dict] = []

        # Try to load FAISS index
        try:
            if not os.path.exists(self.index_path):
                raise FileNotFoundError(self.index_path)
            self.index = faiss.read_index(self.index_path)
        except Exception as e:
            print(f"FAISS index not found at {self.index_path}: {e}")

        # Try to load chunks metadata
        try:
            if not os.path.exists(self.chunks_path):
                raise FileNotFoundError(self.chunks_path)
            if self.chunks_path.lower().endswith(".jsonl"):
                self.chunks = _load_jsonl(self.chunks_path)
            else:
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
        except Exception as e:
            print(f"Chunks file not found at {self.chunks_path}: {e}")

    def retrieve(self, query: str, top_k: int = 3, semester: str = None) -> List[Dict]:
        # If index or chunks are not available, return empty list and log
        if self.index is None:
            print("Cannot retrieve: FAISS index not loaded.")
            return []
        if not self.chunks:
            print("Cannot retrieve: chunks metadata not loaded.")
            return []

        # Extract course and semester from format like "COMPS-Sem-3" -> course="COMPS", semester_num="3"
        course_filter = None
        semester_number = None
        if semester:
            parts = semester.split("-")
            if len(parts) >= 2:
                course_filter = parts[0]  # Get "COMPS" from "COMPS-Sem-3"
                if len(parts) >= 3:
                    semester_number = parts[2]  # Get "3" from "COMPS-Sem-3"
                else:
                    semester_number = parts[1].replace("Sem", "")  # Handle "FY-Sem-1" format

        q_vec = self.embedder.embed_text(query).reshape(1, -1)
        # Search more results initially to allow for filtering
        search_k = top_k * 5 if (course_filter or semester_number) else top_k
        distances, indices = self.index.search(q_vec, search_k)

        results: List[Dict] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            # guard against out-of-range indices
            if idx < 0 or idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            
            # Filter by course if specified (for specialized courses like COMPS, IT, etc.)
            # FY courses should match all since it's common first year
            if course_filter and course_filter != "FY":
                chunk_course = chunk.get("course", "")
                # If chunk doesn't have course field or doesn't match, skip
                if chunk_course and chunk_course != course_filter:
                    continue
            
            # Filter by semester if specified
            if semester_number:
                chunk_semester = chunk.get("semester", "")
                if str(chunk_semester) != semester_number:
                    continue
            
            results.append(chunk)
            
            # Stop once we have enough results
            if len(results) >= top_k:
                break

        return results
