import os
import time
import uuid
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CASE_STUDIES_DIR = Path("./case-studies")
CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "aes-case-studies"


def load_pdfs(directory: Path) -> list:
    documents = []
    pdf_files = sorted(directory.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        documents.extend(docs)
        print(f"  Loaded {len(docs)} pages from {pdf_path.name}")
    return documents


def split_documents(documents: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)
    return [c for c in chunks if c.page_content.strip()]


def _clean_metadata(metadata: dict) -> dict:
    # Chroma only accepts str, int, float, bool — drop None and complex types
    return {
        k: v for k, v in metadata.items()
        if isinstance(v, (str, int, float, bool))
    }


def build_vector_store(chunks: list) -> Chroma:
    # gemini-embedding-001 targets the Gemini Developer API (ai.google.dev).
    # gemini-embedding-2-preview routes to Vertex AI — wrong endpoint for an AI Studio key.
    embedding_fn = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    texts = [c.page_content for c in chunks]
    metadatas = [_clean_metadata(c.metadata) for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    print(f"  Embedding {len(texts)} chunks via Gemini...")
    vectors = []
    for i, text in enumerate(texts):
        vectors.extend(embedding_fn.embed_documents([text]))
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{len(texts)} embedded")
        time.sleep(0.5)   # ~2 req/sec — well within free-tier RPM limit
    print(f"  Received {len(vectors)} embeddings.")

    # Wipe and recreate so re-runs don't accumulate duplicates
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [c.name for c in client.list_collections()] else None
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

    # Wrap in LangChain Chroma so downstream retrieval works normally
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        raise EnvironmentError("GOOGLE_API_KEY is not set. Add it to your .env file.")

    print("=== Step 1: Loading PDFs ===")
    documents = load_pdfs(CASE_STUDIES_DIR)
    print(f"Total pages loaded: {len(documents)}\n")

    print("=== Step 2: Splitting into chunks ===")
    chunks = split_documents(documents)
    print(f"Total chunks created: {len(chunks)}\n")

    print("=== Step 3 & 4: Embedding + storing in Chroma ===")
    build_vector_store(chunks)
    print(f"Vector store persisted at: {CHROMA_PERSIST_DIR}")
    print("\nIngestion complete.")


if __name__ == "__main__":
    main()
