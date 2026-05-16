import os
import sys
import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "aes-case-studies"

QUERIES = [
    "thermal analysis heat management",
    "structural analysis simulation",
    "robotic automation assembly",
]


def get_store() -> Chroma:
    embedding_fn = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


def test_collection_not_empty(store: Chroma) -> bool:
    count = store._collection.count()
    print(f"[collection_not_empty] Documents in store: {count}")
    if count == 0:
        print("  FAIL — collection is empty. Run ingest.py first.")
        return False
    print("  PASS")
    return True


def test_similarity_search(store: Chroma) -> bool:
    print("[similarity_search]")
    passed = True
    for query in QUERIES:
        results = store.similarity_search(query, k=3)
        if not results:
            print(f"  FAIL — no results for: '{query}'")
            passed = False
            continue
        print(f"  query: '{query}'  ->  {len(results)} results")
        for doc in results:
            source = doc.metadata.get("source", "unknown")
            snippet = doc.page_content[:80].replace("\n", " ").encode("ascii", "replace").decode("ascii")
            print(f"    [{source}]  {snippet}...")
    if passed:
        print("  PASS")
    return passed


def test_metadata_present(store: Chroma) -> bool:
    print("[metadata_present]")
    results = store.similarity_search("design engineering", k=5)
    missing = [i for i, d in enumerate(results) if "source" not in d.metadata]
    if missing:
        print(f"  FAIL — {len(missing)} results missing 'source' metadata")
        return False
    print(f"  PASS — all {len(results)} results have 'source' metadata")
    return True


def test_score_threshold(store: Chroma) -> bool:
    print("[score_threshold]")
    results = store.similarity_search_with_relevance_scores(
        "CFD water simulation fenestration", k=3
    )
    if not results:
        print("  FAIL — no results returned")
        return False
    for doc, score in results:
        source = doc.metadata.get("source", "unknown").split("\\")[-1]
        print(f"  score={score:.4f}  [{source}]")
    top_score = results[0][1]
    if top_score < 0.5:
        print(f"  WARN — top relevance score {top_score:.4f} is below 0.5")
    else:
        print("  PASS")
    return True


def main():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY not set.")
        sys.exit(1)

    print("Loading vector store...\n")
    store = get_store()

    results = {
        "collection_not_empty": test_collection_not_empty(store),
        "similarity_search":    test_similarity_search(store),
        "metadata_present":     test_metadata_present(store),
        "score_threshold":      test_score_threshold(store),
    }

    print("\n=== Results ===")
    all_passed = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
