import fitz  # PyMuPDF
import json
import os
from tqdm import tqdm
from pathlib import Path
import re
import os
from pathlib import Path

script_directory = Path(__file__).resolve().parent
os.chdir(script_directory)
print(f"Current working directory: {Path.cwd()}")

# CONFIG
DATA_DIR = "../../data"
OUTPUT_DIR = "output"
CHUNK_SIZE = 500       # approx words
CHUNK_OVERLAP = 50
PROGRESS_FILE = f"{OUTPUT_DIR}/progress.json"
CHUNKS_FILE = f"{OUTPUT_DIR}/chunks.jsonl"
# ----------------------------------------


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def process_pdf(pdf_path, subject, folder_meta, progress):
    doc_key = str(pdf_path)

    last_page_done = progress.get(doc_key, -1)

    doc = fitz.open(pdf_path)
    total_pages = doc.page_count

    print(f"\n📘 Processing {pdf_path.name} ({total_pages} pages)")
    if last_page_done >= 0:
        print(f"↪ Resuming from page {last_page_done + 1}")

    with open(CHUNKS_FILE, "a", encoding="utf-8") as out:
        for page_num in range(last_page_done + 1, total_pages):
            try:
                page = doc.load_page(page_num)
                text = page.get_text()

                if not text or len(text.strip()) < 30:
                    print(
                        f"⚠️  No extractable text on "
                        f"{pdf_path.name} page {page_num + 1} "
                        f"(likely scanned / OCR needed)"
                    )
                    progress[doc_key] = page_num
                    save_progress(progress)
                    continue

                text = clean_text(text)
                chunks = chunk_text(text)

                for chunk in chunks:
                    record = {
                        "text": chunk,
                        "subject": subject,
                        "doc_name": pdf_path.name,
                        "page": page_num + 1,
                        "source_path": str(pdf_path),
                        **folder_meta
                    }
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")

                progress[doc_key] = page_num
                save_progress(progress)

            except Exception as e:
                print(
                    f"❌ ERROR reading {pdf_path.name} page {page_num + 1}\n"
                    f"    {e}\n"
                    f"➡ Progress saved. Re-run to resume."
                )
                save_progress(progress)
                return  # stop this PDF, allow resume

    print(f"✅ Finished {pdf_path.name}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    progress = load_progress()

    for subject_dir in sorted(Path(DATA_DIR).iterdir()):
        if not subject_dir.is_dir():
            continue

        subject = subject_dir.name
        meta_file = subject_dir / "metadata.json"

        if meta_file.exists():
            with open(meta_file, "r") as f:
                folder_meta = json.load(f)
        else:
            folder_meta = {}

        pdf_files = list(subject_dir.glob("*.pdf"))
        if not pdf_files:
            continue

        print(f"\n📂 Subject: {subject}")

        for pdf_path in pdf_files:
            process_pdf(pdf_path, subject, folder_meta, progress)

    print("\n🎉 All files processed")


if __name__ == "__main__":
    main()
