from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.session import get_session_history

_SYSTEM_PROMPT = (
    "You are an expert assistant for AES (Applied Engineering Solutions), "
    "a company specialising in engineering design, simulation, and analysis. "
    "Answer questions based strictly on the following case studies and documents. "
    "If the answer is not found in the context, say you don't have that information.\n\n"
    "Context:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(persist_directory: str):
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="aes-case-studies",
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", streaming=True)

    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    async def _retrieve(x: dict) -> str:
        docs = await retriever.ainvoke(x["question"])
        return format_docs(docs)

    rag_chain = (
        RunnablePassthrough.assign(context=_retrieve)
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
