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
DEFAULT_INDEX_PATH = OUTPUT_DIR / "FY_Sem-1_faiss.index"

# Resume/checkpointing
PROGRESS_FILE = OUTPUT_DIR / "progress_faiss"
RESUME_FROM_PROGRESS = True
SAVE_INDEX_EVERY_BATCH = True

# Resume behavior:
# - If progress_faiss == 0: rebuild from scratch (overwrite index on save)
# - If progress_faiss > 0: must load existing index and append
REBUILD_IF_PROGRESS_ZERO = True

# Throttle embedding calls to avoid rate limits
EMBED_REQUESTS_PER_SECOND = 1.4
MIN_SECONDS_BETWEEN_EMBEDS = 1.0 / EMBED_REQUESTS_PER_SECOND

# Retry handling for quota/rate limits
MAX_EMBED_RETRIES = 10
DEFAULT_RETRY_SLEEP_SECONDS = 12.0

# Print progress so you can see it working
PRINT_EVERY_N = 1

# For quick testing: ingest only first N chunks.
# Set INGEST_LIMIT=0 to ingest everything.
DEFAULT_INGEST_LIMIT = 0


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


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def read_progress(path: Path = PROGRESS_FILE) -> int:
    try:
        raw = path.read_text(encoding="utf-8").strip()
        return int(raw) if raw else 0
    except FileNotFoundError:
        return 0
    except Exception:
        return 0


def write_progress(value: int, path: Path = PROGRESS_FILE) -> None:
    _atomic_write_text(path, str(int(value)))


def save_faiss_index(index: faiss.Index, path: Path) -> None:
    # Write atomically to avoid corrupting the main index on crashes.
    tmp = Path(str(path) + ".tmp")
    faiss.write_index(index, str(tmp))
    os.replace(tmp, path)


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

        # Resume: decide whether to rebuild or append based on progress_faiss.
        resume_from = read_progress() if RESUME_FROM_PROGRESS else 0

        index: Optional[faiss.Index] = None
        if RESUME_FROM_PROGRESS and resume_from <= 0 and REBUILD_IF_PROGRESS_ZERO:
            # Explicit rebuild from scratch.
            if self.index_path.exists():
                print(
                    f"Rebuilding from scratch (progress_faiss=0). Existing index will be overwritten: {self.index_path}"
                )
            resume_from = 0

        elif RESUME_FROM_PROGRESS and resume_from > 0:
            # Must load existing index and append.
            if not self.index_path.exists():
                raise RuntimeError(
                    f"progress_faiss={resume_from} but index file is missing: {self.index_path}"
                )

            try:
                index = faiss.read_index(str(self.index_path))
            except Exception as e:
                raise RuntimeError(f"Failed to read existing index at {self.index_path}: {e}") from e

            existing = int(getattr(index, "ntotal", 0))
            if existing <= 0:
                print(
                    f"⚠ progress_faiss={resume_from} but loaded index.ntotal={existing}. "
                    f"Treating as empty index and rebuilding from scratch."
                )
                index = None
                resume_from = 0
                write_progress(0)
            elif existing != resume_from:
                # Avoid duplicates: the real source of truth is what's inside the index.
                print(
                    f"⚠ progress_faiss={resume_from} but index.ntotal={existing}; "
                    f"resuming from index.ntotal to avoid duplicates"
                )
                resume_from = existing

        else:
            # Resume disabled or progress missing: start from scratch.
            resume_from = 0

        if resume_from > 0:
            print(f"Resuming FAISS build from vector #{resume_from} (already indexed)")

        vectors_batch: list[np.ndarray] = []
        total = 0

        last_embed_at: Optional[float] = None

        # Cursor counts how many valid chunks (with text) we've passed.
        cursor = 0

        # Build vectors in the SAME order as chunks.jsonl lines.
        fatal_error: Optional[Exception] = None

        def commit_batch() -> None:
            nonlocal index
            if index is None or not vectors_batch:
                return
            index.add(np.vstack(vectors_batch))
            vectors_batch.clear()

            if SAVE_INDEX_EVERY_BATCH:
                save_faiss_index(index, self.index_path)
                write_progress(int(index.ntotal))
                print(f"✅ Saved checkpoint: {self.index_path.name} (ntotal={int(index.ntotal)})")

        for record in iter_jsonl(self.input_jsonl):
            text = record.get("text")
            if not text:
                continue

            # Skip already-committed chunks when resuming.
            if resume_from > 0 and cursor < resume_from:
                cursor += 1
                continue

            if limit > 0 and total >= limit:
                print(f"Reached INGEST_LIMIT={limit}. Stopping early.")
                break

            cursor += 1

            # Print progress so you know it's alive
            if PRINT_EVERY_N > 0 and (total % PRINT_EVERY_N == 0):
                doc_name = record.get("doc_name") or record.get("doc") or "(unknown)"
                page = record.get("page")
                extra = f" p{page}" if page is not None else ""
                overall_i = resume_from + total + 1
                overall_total = total_lines
                if limit > 0:
                    run_total = min(limit, max(0, total_lines - resume_from))
                    print(
                        f"Embedding {overall_i}/{overall_total} (run {total + 1}/{run_total}): {doc_name}{extra}"
                    )
                else:
                    print(f"Embedding {overall_i}/{overall_total}: {doc_name}{extra}")

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
                        fatal_error = e
                        break
                    wait_s = extract_retry_after_seconds(e)
                    attempt += 1
                    print(
                        f"Rate-limited (attempt {attempt}/{MAX_EMBED_RETRIES}). "
                        f"Sleeping {wait_s:.1f}s..."
                    )
                    time.sleep(wait_s)

            if fatal_error is not None:
                print(f"❌ Embedding failed. Saving current progress then exiting. Error: {fatal_error}")
                # Save what we already have in-memory.
                commit_batch()
                # Also save index even if the last batch isn't full.
                if index is not None and vectors_batch:
                    index.add(np.vstack(vectors_batch))
                    vectors_batch.clear()
                    save_faiss_index(index, self.index_path)
                    write_progress(int(index.ntotal))
                    print(f"✅ Saved final partial batch (ntotal={int(index.ntotal)})")
                return int(index.ntotal) if index is not None else 0

            if index is None:
                index = faiss.IndexFlatL2(int(vec.shape[0]))

            vectors_batch.append(vec)
            total += 1

            if len(vectors_batch) >= self.batch_size:
                commit_batch()
                print(f"Added batch to FAISS. Total embedded so far (this run): {total}")

        if index is None:
            print("No valid records found to ingest.")
            return 0

        if vectors_batch:
            index.add(np.vstack(vectors_batch))
            vectors_batch.clear()

        # Final save
        save_faiss_index(index, self.index_path)
        write_progress(int(index.ntotal))

        print(f"Ingestion complete: {total} chunks (this run)")
        print(f"Wrote index: {self.index_path} (ntotal={int(index.ntotal)})")
        print(f"Using chunks metadata from: {self.input_jsonl}")
        return int(index.ntotal)


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
