"""
document_loader.py

Job: Read every policy .txt file from policy_docs/, and split each one into
small chunks (not whole documents). We chunk because embeddings work
better on focused pieces of text, not entire multi-topic documents.

Chunking strategy: one logical unit per chunk - each numbered policy
section, or each FAQ Q&A pair, becomes its own standalone chunk. We do
NOT pack multiple sections together to hit a target size, because that
dilutes the embedding of the one chunk that should be the strongest
match for a given question (e.g. mixing the "Return Window" section with
an unrelated "Refund Processing Time" section makes both harder to find).

The document header (title / "Document Owner:" / "Last Updated:" lines)
is dropped before chunking - it's pure metadata with no customer-facing
answer content, and including it was diluting the embedding of whichever
section it got attached to.

A single section that's unusually long is still split on sentence
boundaries as a fallback - never mid-word.
"""

import os
import re


def load_documents(folder_path):
    """Reads every .txt file in the folder. Returns a list of dicts:
    {source: filename, text: full file content}"""
    documents = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"source": filename, "text": text})
    return documents


def _is_header_paragraph(paragraph):
    """Detects the document metadata header (title, owner, last-updated
    lines) so it can be dropped before chunking - it's boilerplate, not
    answer content, and pollutes whichever section it gets attached to."""
    return "Document Owner:" in paragraph and "Last Updated:" in paragraph


def _split_oversized_paragraph(paragraph, target_size):
    """Fallback for a single section longer than target_size: splits on
    sentence boundaries (never mid-word) and groups sentences up to
    target_size. Only used if a section is unusually long."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= target_size or not current:
            current = candidate
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text, target_size=500):
    """Splits one document's text into chunks aligned to paragraph
    boundaries (blank-line-separated sections/Q&A pairs). Each section
    becomes its own chunk - no merging across section boundaries, so no
    chunk's embedding gets diluted by unrelated neighboring content. The
    header/metadata paragraph is dropped. A section far larger than
    target_size is sentence-split as a fallback."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    for para in paragraphs:
        if _is_header_paragraph(para):
            continue
        if len(para) > target_size * 1.5:
            chunks.extend(_split_oversized_paragraph(para, target_size))
        else:
            chunks.append(para)

    return chunks


def load_and_chunk_all(folder_path):
    """Combines loading + chunking. Returns a list of dicts:
    {source: filename, chunk_id: index, text: chunk text}
    This is the final format the embedding step will use."""
    documents = load_documents(folder_path)
    all_chunks = []
    for doc in documents:
        chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": chunk
            })
    return all_chunks


if __name__ == "__main__":
    # Quick manual test: run this file directly to see chunks printed
    chunks = load_and_chunk_all("../../policy_docs")
    print(f"Loaded {len(chunks)} chunks from policy_docs/")
    for c in chunks:
        if c["source"] in ("02_returns_policy.txt", "03_website_faq.txt"):
            print(f"\n[{c['source']} #{c['chunk_id']}] ({len(c['text'])} chars)")
            print(c["text"])
