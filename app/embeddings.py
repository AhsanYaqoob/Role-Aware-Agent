from langchain_core.embeddings import Embeddings
from fastembed import TextEmbedding


class FastEmbedEmbeddings(Embeddings):
    """Adapts fastembed's ONNX-based TextEmbedding to LangChain's Embeddings
    interface, so FAISS can use it without depending on torch."""

    def __init__(self, model_name: str):
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [vec.tolist() for vec in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()
