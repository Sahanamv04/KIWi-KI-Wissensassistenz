import csv
import re
import config
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
import ollama

# import pymupdf  # PDF ingestion is intentionally inactive.


def clean(text):
    text = text.replace("\u00ad", "")                # remove invisible soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)     # join words split across lines
    text = re.sub(r"[ \t]+", " ", text)              # collapse repeated spaces
    text = re.sub(r" ?\n ?", "\n", text)             # strip spaces around line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)           # collapse many blank lines
    # PDF-only cleanup, retained here as comments for reference:
    # text = re.sub(r"(?im)^focus\s+funding\s+\w+\s+\d{4}(\s+\d{1,3})?\s*$", "", text)
    # text = re.sub(r"(?m)^\d{1,3}$", "", text)
    return text.strip()


# PDF ingestion is no longer active; CSV records are loaded in build_chunks().
# def load_pages(path):
#     with pymupdf.open(path) as doc:
#         for i, page in enumerate(doc, start=1):
#             text = clean(page.get_text("text"))
#             if len(text) > 40:
#                 yield i, text


def build_chunks():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = {}
    total = 0
    records = 0
    csv_files = sorted(config.DATA_DIR.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {config.DATA_DIR}")

    fields = (
        ("Question", "question"),
        ("Answer", "answer"),
        ("Key fact", "key_fact"),
        ("Evidence", "evidence_text"),
        ("Topic", "topic"),
        ("Section", "section"),
        ("Audience", "audience"),
        ("Portal path", "portal_path"),
        ("Contact route", "contact_route"),
        ("Keywords", "keywords"),
        ("Source", "source_id"),
        ("Source section", "source_section_heading"),
        ("Source URL", "source_url"),
        ("Verified", "source_last_verified"),
    )

    for csv_file in csv_files:
        with csv_file.open(encoding="utf-8-sig", newline="") as file:
            for row_number, row in enumerate(csv.DictReader(file), start=1):
                text = clean("\n".join(
                    f"{label}: {value.strip()}"
                    for label, column in fields
                    if (value := row.get(column, "") or "").strip()
                ))
                if not text:
                    continue

                records += 1
                record_id = (
                    row.get("record_id", "").strip()
                    or row.get("chunk_id", "").strip()
                    or f"{csv_file.stem}-{row_number}"
                )
                source = row.get("source_id", "").strip() or csv_file.name
                section = (
                    row.get("section", "").strip()
                    or row.get("source_section_heading", "").strip()
                )

                for chunk_number, piece in enumerate(splitter.split_text(text)):
                    total += 1
                    chunk_id = hashlib.sha1(
                        f"{csv_file.name}:{record_id}:{chunk_number}:{piece}".encode()
                    ).hexdigest()
                    chunks[chunk_id] = {
                        "text": piece,
                        "source": source,
                        "record_id": record_id,
                        "section": section,
                        "source_url": row.get("source_url", "").strip(),
                        "source_last_verified": row.get("source_last_verified", "").strip(),
                    }

    return chunks, total, records


# Previous PDF-only indexing loop:
# for pdf in sorted(config.DATA_DIR.glob("**/*.pdf")):
#     for page_no, text in load_pages(pdf):
#         for piece in splitter.split_text(text):
#             chunk_id = hashlib.sha1(piece.encode()).hexdigest()
#             chunks[chunk_id] = {"text": piece, "source": pdf.name,
#                                 "page": page_no, "also_in": []}


def build_index():
    chunks, total, records = build_chunks()
    print(f"{total} chunks from {records} CSV records")

    client = ollama.Client(host=config.OLLAMA_HOST)
    db = chromadb.PersistentClient(path=config.CHROMA_DIR)
    try:
        db.delete_collection(config.COLLECTION)   # rebuild from scratch every run
    except Exception:
        pass
    col = db.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    ids = list(chunks.keys())
    B = 32                                         # embed 32 chunks per request
    for s in range(0, len(ids), B):
        batch_ids = ids[s:s + B]
        batch = [chunks[i] for i in batch_ids]
        emb = client.embed(model=config.EMBED_MODEL,
                           input=[c["text"] for c in batch])["embeddings"]
        col.add(
            ids=batch_ids,
            embeddings=emb,
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "record_id": c["record_id"],
                        "section": c["section"], "source_url": c["source_url"],
                        "source_last_verified": c["source_last_verified"]}
                       for c in batch],
        )
        print(f"indexed {min(s + B, len(ids))}/{len(ids)}")
    print("Done. Stored in ChromaDB:", col.count(), "chunks")


if __name__ == "__main__":
    build_index()