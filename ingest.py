import os
import glob

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "data"
VECTORSTORE_DIR = "vectorstore"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_documents():
    documents = []
    for file_path in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        documents.extend(TextLoader(file_path, encoding="utf-8").load())
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def main():
    print("Loading documents from data/ ...")
    documents = load_documents()
    print(f"Loaded {len(documents)} document(s).")

    print("Splitting into chunks (size=500, overlap=50, prefers paragraph/sentence breaks)...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunk(s).")

    print(f"Embedding chunks with {EMBEDDING_MODEL} ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building FAISS index and saving to vectorstore/ ...")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_DIR)

    print(f"Done. Saved FAISS index to '{VECTORSTORE_DIR}/'.")


if __name__ == "__main__":
    main()
