from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from fastembed.rerank.cross_encoder import TextCrossEncoder

from app.embeddings import FastEmbedEmbeddings

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

_embeddings = FastEmbedEmbeddings(EMBEDDING_MODEL)
_vectorstore = FAISS.load_local(
    VECTORSTORE_DIR, _embeddings, allow_dangerous_deserialization=True
)
_reranker = TextCrossEncoder(model_name=RERANKER_MODEL)


def get_candidate_texts(query: str, k: int = 10) -> list[str]:
    docs = _vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in docs]


def rerank_texts(query: str, texts: list[str], top_k: int = 3) -> list[str]:
    if not texts:
        return []
    scores = list(_reranker.rerank(query, texts))
    ranked = sorted(zip(scores, texts), key=lambda pair: pair[0], reverse=True)
    return [text for _, text in ranked[:top_k]]


def retrieve_and_rerank(query: str) -> str:
    candidates = get_candidate_texts(query, k=10)
    top_chunks = rerank_texts(query, candidates, top_k=3)
    return "\n\n".join(top_chunks)
