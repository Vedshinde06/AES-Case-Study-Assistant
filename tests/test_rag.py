from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from app.rag import format_docs, build_chain


def test_format_docs_joins_with_double_newline():
    docs = [Document(page_content="Alpha"), Document(page_content="Beta")]
    result = format_docs(docs)
    assert result == "Alpha\n\nBeta"


def test_format_docs_single_doc():
    docs = [Document(page_content="Only one")]
    assert format_docs(docs) == "Only one"


def test_format_docs_empty_list():
    assert format_docs([]) == ""


def test_build_chain_returns_runnable():
    mock_vectorstore = MagicMock()
    mock_vectorstore.as_retriever.return_value = MagicMock()

    with patch("app.rag.GoogleGenerativeAIEmbeddings") as mock_emb_cls, \
         patch("app.rag.Chroma") as mock_chroma_cls, \
         patch("app.rag.ChatGoogleGenerativeAI"):

        mock_chroma_cls.return_value = mock_vectorstore
        chain = build_chain("./fake_db")

    assert chain is not None
    mock_chroma_cls.assert_called_once_with(
        persist_directory="./fake_db",
        embedding_function=mock_emb_cls.return_value,
        collection_name="aes-case-studies",
    )
    mock_vectorstore.as_retriever.assert_called_once_with(search_kwargs={"k": 5})
