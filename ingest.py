import os
import re
import glob
import pickle

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from app.embeddings import FastEmbedEmbeddings

load_dotenv()

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_SECTION_CHARS = 60

_SECTION_HEADER_RE = re.compile(r"\n(?=\d+\.(?:\d+)?\s)")

_sub_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _split_into_sections(text: str) -> list[str]:
    """Splits on numbered section headers (e.g. '2.1 Total Revenue') so a
    heading never gets separated from its own content. A short top-level
    header with no body of its own (e.g. '2. REVENUE' immediately followed
    by '2.1 ...') gets merged forward into the section that follows it,
    instead of becoming its own near-empty chunk."""
    raw_pieces = [p for p in _SECTION_HEADER_RE.split(text) if p.strip()]
    sections = []
    buffer = ""
    for i, piece in enumerate(raw_pieces):
        buffer = f"{buffer}\n{piece}" if buffer else piece
        is_last = i == len(raw_pieces) - 1
        if len(buffer.strip()) >= MIN_SECTION_CHARS or is_last:
            sections.append(buffer.strip())
            buffer = ""
    return sections


def _section_label(section_text: str) -> str:
    return section_text.splitlines()[0].strip()


def load_and_chunk_documents() -> list[Document]:
    chunks = []
    for file_path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        source = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        for section_text in _split_into_sections(text):
            label = _section_label(section_text)
            if len(section_text) <= CHUNK_SIZE:
                chunks.append(Document(page_content=section_text, metadata={"source": source, "section": label}))
            else:
                # A too-long section gets split further, but the header must
                # never end up alone in its own chunk (the plain splitter
                # will do that if header + body barely exceeds CHUNK_SIZE) --
                # so split only the body, then force-attach the header to the
                # first piece even if that pushes it slightly over budget.
                header, _, body = section_text.partition("\n")
                body = body.strip()
                pieces = _sub_splitter.split_text(body) if body else [""]
                for idx, piece in enumerate(pieces):
                    content = f"{header}\n{piece}" if idx == 0 else piece
                    chunks.append(Document(page_content=content, metadata={"source": source, "section": label}))
    return chunks


def main():
    print("Loading and chunking documents from data/ (section-aware boundaries)...")
    chunks = load_and_chunk_documents()
    print(f"Created {len(chunks)} chunk(s).")

    print(f"Embedding chunks with {EMBEDDING_MODEL} ...")
    embeddings = FastEmbedEmbeddings(EMBEDDING_MODEL)

    print("Building FAISS index and saving to vectorstore/ ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    chunks_path = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunk(s) with metadata to '{chunks_path}' (used for BM25 hybrid search).")

    print(f"Done. Saved FAISS index + chunk metadata to '{VECTORSTORE_DIR}/'.")


if __name__ == "__main__":
    main()
