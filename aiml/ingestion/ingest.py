import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

from embedder import Embedder


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"  # one folder above: aiml/.env
OUTPUT_DIR = SCRIPT_DIR / "output"

DEFAULT_INPUT_JSONL = OUTPUT_DIR / "chunks.jsonl"
DEFAULT_INDEX_PATH = OUTPUT_DIR / "faiss.index"

# Throttle embedding calls to avoid rate limits
EMBED_REQUESTS_PER_SECOND = 1.0
MIN_SECONDS_BETWEEN_EMBEDS = 1.0 / EMBED_REQUESTS_PER_SECOND

# Retry handling for quota/rate limits
MAX_EMBED_RETRIES = 10
DEFAULT_RETRY_SLEEP_SECONDS = 12.0

# Print progress so you can see it working
PRINT_EVERY_N = 1

# For quick testing: ingest only first N chunks.
# Set INGEST_LIMIT=0 to ingest everything.
DEFAULT_INGEST_LIMIT = 50


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {line_num} in {path}: {e}") from e


def count_non_empty_lines(path: Path) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc)
    return (
        ("429" in msg)
        or ("RESOURCE_EXHAUSTED" in msg)
        or ("rate" in msg.lower() and "limit" in msg.lower())
        or ("quota" in msg.lower() and "exceed" in msg.lower())
    )


def extract_retry_after_seconds(exc: Exception) -> float:
    # The SDK error often includes: "Please retry in 10.150251921s."
    msg = str(exc)
    marker = "retry in "
    if marker in msg:
        tail = msg.split(marker, 1)[1]
        num = ""
        for ch in tail:
            if ch.isdigit() or ch == ".":
                num += ch
            elif ch.lower() == "s":
                break
            elif num:
                break
        try:
            if num:
                return float(num)
        except Exception:
            pass
    return DEFAULT_RETRY_SLEEP_SECONDS


class Ingestor:
    def __init__(
        self,
        embedder: Embedder,
        input_jsonl: Path = DEFAULT_INPUT_JSONL,
        index_path: Path = DEFAULT_INDEX_PATH,
        batch_size: int = 32,
    ):
        self.embedder = embedder
        self.input_jsonl = input_jsonl
        self.index_path = index_path
        self.batch_size = batch_size

    def ingest(self) -> int:
        if not self.input_jsonl.exists():
            print(f"Input chunks file not found: {self.input_jsonl}")
            return 0

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        total_lines = count_non_empty_lines(self.input_jsonl)
        limit_env = os.getenv("INGEST_LIMIT")
        try:
            limit = int(limit_env) if limit_env is not None else DEFAULT_INGEST_LIMIT
        except ValueError:
            limit = DEFAULT_INGEST_LIMIT

        if limit <= 0:
            effective_total = total_lines
            print(f"Found {total_lines} chunk lines in {self.input_jsonl} (no limit)")
        else:
            effective_total = min(total_lines, limit)
            print(
                f"Found {total_lines} chunk lines in {self.input_jsonl} "
                f"(limiting to first {effective_total}; set INGEST_LIMIT=0 for all)"
            )

        index: Optional[faiss.Index] = None
        vectors_batch: list[np.ndarray] = []
        total = 0

        last_embed_at: Optional[float] = None

        # Build vectors in the SAME order as chunks.jsonl lines.
        for record in iter_jsonl(self.input_jsonl):
            text = record.get("text")
            if not text:
                continue

            if limit > 0 and total >= limit:
                print(f"Reached INGEST_LIMIT={limit}. Stopping early.")
                break

            # Print progress so you know it's alive
            if PRINT_EVERY_N > 0 and (total % PRINT_EVERY_N == 0):
                doc_name = record.get("doc_name") or record.get("doc") or "(unknown)"
                page = record.get("page")
                extra = f" p{page}" if page is not None else ""
                print(f"Embedding {total + 1}/{effective_total}: {doc_name}{extra}")

            # Enforce 1 request / second
            now = time.monotonic()
            if last_embed_at is not None:
                elapsed = now - last_embed_at
                if elapsed < MIN_SECONDS_BETWEEN_EMBEDS:
                    time.sleep(MIN_SECONDS_BETWEEN_EMBEDS - elapsed)

            # Retry on quota/rate-limit errors
            attempt = 0
            while True:
                try:
                    vec = self.embedder.embed_text(text)
                    last_embed_at = time.monotonic()
                    break
                except Exception as e:
                    if not is_rate_limit_error(e) or attempt >= MAX_EMBED_RETRIES:
                        raise
                    wait_s = extract_retry_after_seconds(e)
                    attempt += 1
                    print(
                        f"Rate-limited (attempt {attempt}/{MAX_EMBED_RETRIES}). "
                        f"Sleeping {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)
            if index is None:
                index = faiss.IndexFlatL2(int(vec.shape[0]))

            vectors_batch.append(vec)
            total += 1

            if len(vectors_batch) >= self.batch_size:
                index.add(np.vstack(vectors_batch))
                vectors_batch.clear()
                print(f"Added batch to FAISS. Total embedded so far: {total}")

        if index is None:
            print("No valid records found to ingest.")
            return 0

        if vectors_batch:
            index.add(np.vstack(vectors_batch))

        faiss.write_index(index, str(self.index_path))
        print(f"Ingestion complete: {total} chunks")
        print(f"Wrote index: {self.index_path}")
        print(f"Using chunks metadata from: {self.input_jsonl}")
        return total


def main() -> None:
    # Load env from one folder above this script (aiml/.env)
    load_dotenv(dotenv_path=ENV_PATH)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(f"GEMINI_API_KEY is not set (expected in {ENV_PATH})")

    client = genai.Client(api_key=api_key)
    embedder = Embedder(client=client)
    ingestor = Ingestor(embedder=embedder)
    ingestor.ingest()


if __name__ == "__main__":
    main()
