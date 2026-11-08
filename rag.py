
import chromadb
import ollama
import config

client = ollama.Client(host=config.OLLAMA_HOST)

collection = chromadb.PersistentClient(
    path=config.CHROMA_DIR
).get_collection(config.COLLECTION)

CSV_METADATA_FIELDS = {
    "source",
    "record_id",
    "section",
    "source_url",
    "source_last_verified",
}


def validate_csv_metadata(metadata):
    missing = CSV_METADATA_FIELDS - metadata.keys()
    if missing:
        raise RuntimeError(
            "The ChromaDB index is from the previous data format and cannot be "
            "used with CSV data. Rebuild it by running `python ingest.py`, then "
            "restart with `python rag.py`."
        )


def retrieve(query, k=config.TOP_K):
    q_emb = client.embed(
        model=config.EMBED_MODEL,
        input=[query]
    )["embeddings"][0]

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=k * 4
    )

    hits, seen = [], set()

    for doc, meta, dist in zip(
        res["documents"][0],
        res["metadatas"][0],
        res["distances"][0]
    ):
        validate_csv_metadata(meta)
        key = " ".join(doc.split())[:150].lower()

        if key in seen:
            continue

        seen.add(key)

        hits.append({
            "text": doc,
            "source": meta["source"],
            "record_id": meta["record_id"],
            "section": meta["section"],
            "source_url": meta["source_url"],
            "source_last_verified": meta["source_last_verified"],
            "distance": dist
        })

        if len(hits) == k:
            break

    return hits


def format_source(hit):
    parts = [hit["source"], f"record {hit['record_id']}"]
    if hit["section"]:
        parts.append(hit["section"])
    if hit["source_last_verified"]:
        parts.append(f"verified {hit['source_last_verified']}")
    if hit["source_url"]:
        parts.append(hit["source_url"])
    return "; ".join(parts)


def add_source_link(text, hits):
    if any(hit["source_url"] and hit["source_url"] in text for hit in hits):
        return text

    for hit in hits:
        if hit["source_url"]:
            label = hit["section"] or hit["source"]
            return f"{text.rstrip()}\n\n[{label}]({hit['source_url']})"
    return text


SYSTEM_PROMPT = """You are KIWi, a knowledge assistant for THWS staff.
Answer ONLY from the numbered context passages below.
- Cite the passages you used like [1], [2].
- If the context does not contain the answer, say you could not find it in the data. Do not guess.
- If passages conflict, say so and prefer the most recently verified source when dates are available.
- Answer in the same language as the question.
- Give a clear, useful explanation rather than a one-line summary. Include relevant steps, conditions, dates, contact details, and next actions when the context provides them; do not pad the answer or invent details.
- Include a direct Markdown link to the most relevant source website in the answer, using its URL from the context."""


def build_prompt(query, hits):
    parts = []

    for i, h in enumerate(hits, 1):
        parts.append(
            f"[{i}] (source: {format_source(h)})\n{h['text']}"
        )

    context = "\n\n".join(parts)

    return f"Context:\n{context}\n\nQuestion: {query}"


def answer(query, k=config.TOP_K):
    hits = retrieve(query, k)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_prompt(query, hits)}
    ]

    resp = client.chat(
        model=config.LLM_MODEL,
        messages=messages,
        options={"temperature": 0.1}
    )

    text = add_source_link(resp["message"]["content"], hits)
    return text, hits


if __name__ == "__main__":
    print("KIWi - THWS Knowledge Assistant")
    print("Ask a question or type 'exit' to quit.\n")

    while True:
        try:
            query = input("You: ").strip()

            if query.lower() in ("exit", "quit"):
                print("KIWi: Goodbye!")
                break

            if not query:
                continue

            text, hits = answer(query)

            print("KIWi:", text)

            if hits:
                print("\nSources:")

                for i, h in enumerate(hits, 1):
                    print(
                        f"  [{i}] {format_source(h)} "
                        f"distance={h['distance']:.3f}"
                    )
            else:
                print("\nNo relevant document passages were retrieved.")

            print()

        except KeyboardInterrupt:
            print("\nKIWi: Goodbye!")
            break

        except Exception as e:
            print(f"\nAn error occurred: {e}\n")
