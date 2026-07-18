import pickle
import re

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rank_bm25 import BM25Okapi

from app.embeddings import FastEmbedEmbeddings

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


_embeddings = FastEmbedEmbeddings(EMBEDDING_MODEL)
_vectorstore = FAISS.load_local(
    VECTORSTORE_DIR, _embeddings, allow_dangerous_deserialization=True
)
_reranker = TextCrossEncoder(model_name=RERANKER_MODEL)

with open(f"{VECTORSTORE_DIR}/chunks.pkl", "rb") as _f:
    _all_chunks: list[Document] = pickle.load(_f)
_bm25 = BM25Okapi([_tokenize(doc.page_content) for doc in _all_chunks])


def _bm25_search(query: str, k: int) -> list[Document]:
    scores = _bm25.get_scores(_tokenize(query))
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_all_chunks[i] for i in ranked_idx]


def _reciprocal_rank_fusion(rankings: list[list[Document]], k: int = RRF_K) -> list[Document]:
    """Merges multiple ranked lists (semantic + keyword search) into one
    ranking. Each document's fused score is the sum of 1/(k + rank) across
    every list it appears in, so a document ranked highly by BOTH methods
    rises to the top -- without needing to normalize each method's raw
    scores (cosine similarity vs. BM25 term-frequency) onto a shared scale."""
    scores: dict[str, float] = {}
    lookup: dict[str, Document] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            key = doc.page_content
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            lookup[key] = doc
    ordered_keys = sorted(scores, key=lambda kk: scores[kk], reverse=True)
    return [lookup[kk] for kk in ordered_keys]


def get_candidate_documents(query: str, k: int = 10) -> list[Document]:
    """Hybrid retrieval: FAISS semantic search catches paraphrases and
    conceptual matches; BM25 keyword search catches exact numbers, codes,
    and terms that embeddings can blur together. Merged via Reciprocal Rank
    Fusion instead of picking one or the other."""
    semantic = _vectorstore.similarity_search(query, k=k)
    keyword = _bm25_search(query, k=k)
    fused = _reciprocal_rank_fusion([semantic, keyword])
    return fused[:k]


def rerank_documents(query: str, docs: list[Document], top_k: int = 3) -> list[Document]:
    if not docs:
        return []
    texts = [d.page_content for d in docs]
    scores = list(_reranker.rerank(query, texts))
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def format_context(docs: list[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        section = doc.metadata.get("section", "")
        label = f"{source}, {section}" if section else source
        parts.append(f"[Source: {label}]\n{doc.page_content}")
    return "\n\n".join(parts)
