from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from dotenv import load_dotenv
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from qdrant_client import QdrantClient, models


class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    context: list[Document]


class UniCrawlerVectorStore(QdrantVectorStore):
    """Custom QdrantVectorStore that handles UniCrawler root-level metadata."""

    @classmethod
    def _document_from_point(
        cls,
        scored_point: Any,
        collection_name: str,
        content_payload_key: str,
        metadata_payload_key: str,
    ) -> Document:
        # UniCrawler does not nest metadata. If metadata_payload_key is not found,
        # we fallback to using the entire payload as metadata.
        payload = scored_point.payload or {}
        metadata = payload.get(metadata_payload_key) if metadata_payload_key else None

        if metadata is None:
            # Fallback: use root payload as metadata
            metadata = payload.copy()
        elif not isinstance(metadata, dict):
            # If it's not a dict, we can't easily merge, but let's try to keep it
            metadata = {metadata_payload_key: metadata}

        metadata["_id"] = scored_point.id
        metadata["_collection_name"] = collection_name
        return Document(
            page_content=payload.get(content_payload_key, ""),
            metadata=metadata,
        )


def get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and not value:
        raise ValueError(f"Missing required environment variable: {name}")
    if value is None:
        raise ValueError(f"Missing environment variable: {name}")
    return value


def build_embeddings() -> FastEmbedEmbeddings:
    provider = get_env("EMBEDDING_PROVIDER", "fastembed").lower()
    if provider != "fastembed":
        raise ValueError(
            "Unsupported EMBEDDING_PROVIDER. Supported: fastembed. "
            "Set EMBEDDING_PROVIDER=fastembed to use local FastEmbed."
        )
    model_name = get_env("FASTEMBED_MODEL", "BAAI/bge-small-en-v1.5")
    return FastEmbedEmbeddings(model_name=model_name)


def get_collection_name() -> str:
    domain = os.getenv("QDRANT_DOMAIN")
    if domain:
        # Format: unicrawler_<dominio_senza_punti>
        clean_domain = domain.replace(".", "_")
        return f"unicrawler_{clean_domain}"
    return get_env("QDRANT_COLLECTION", "arag")


def build_retriever() -> BaseRetriever:
    qdrant_url = get_env("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = get_collection_name()
    content_key = get_env("QDRANT_TEXT_KEY", "text")
    metadata_key = os.getenv("QDRANT_METADATA_KEY", "metadata")
    top_k = int(get_env("TOP_K", "4"))

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Check connection early to provide better error message
    try:
        client.get_collections()
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to Qdrant at {qdrant_url}. "
            "Ensure Qdrant is running and the URL is correct."
        ) from e

    # Optional filtering
    filter_key = os.getenv("QDRANT_FILTER_KEY")
    filter_value = os.getenv("QDRANT_FILTER_VALUE")
    qdrant_filter = None
    if filter_key and filter_value:
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=filter_key,
                    match=models.MatchValue(value=filter_value),
                )
            ]
        )

    vectorstore = UniCrawlerVectorStore(
        client=client,
        collection_name=collection,
        embedding=build_embeddings(),
        content_payload_key=content_key,
        metadata_payload_key=metadata_key,
    )

    search_kwargs = {"k": top_k}
    if qdrant_filter:
        search_kwargs["filter"] = qdrant_filter

    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def format_docs(docs: list[Document]) -> str:
    if not docs:
        return "No relevant context found."
    formatted = []
    for index, doc in enumerate(docs, start=1):
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("url")
            or doc.metadata.get("file")
        )
        header = f"[{index}] {source}".strip() if source else f"[{index}]"
        formatted.append(f"{header}\n{doc.page_content}")
    return "\n\n".join(formatted)


def latest_user_message(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if message.type == "human":
            return message.content
    raise ValueError("No user message found in conversation state.")


def build_graph():
    retriever = build_retriever()
    llm = ChatGroq(
        api_key=get_env("GROQ_API_KEY", required=True),
        model=get_env("GROQ_MODEL", "llama-3.1-8b-instant"),
        temperature=float(get_env("GROQ_TEMPERATURE", "0.2")),
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an ARAG terminal assistant. Answer the user using only the "
                "retrieved context. If the answer is not in the context, say you do not know.",
            ),
            ("system", "Context:\n{context}"),
            MessagesPlaceholder("messages"),
        ]
    )
    chain = prompt | llm

    def retrieve(state: GraphState) -> dict[str, list[Document]]:
        question = latest_user_message(state["messages"])
        docs = retriever.invoke(question)
        return {"context": docs}

    def generate(state: GraphState) -> dict[str, list[BaseMessage]]:
        context = format_docs(state.get("context", []))
        response = chain.invoke({"context": context, "messages": state["messages"]})
        return {"messages": [response]}

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")
    graph_builder.add_edge("generate", END)
    return graph_builder.compile(checkpointer=MemorySaver())


def print_sources(docs: list[Document]) -> None:
    if not docs:
        print("Sources: none")
        return
    print("Sources:")
    for index, doc in enumerate(docs, start=1):
        source = (
            doc.metadata.get("source")
            or doc.metadata.get("url")
            or doc.metadata.get("file")
        )
        label = source or "unknown"
        print(f"  {index}. {label}")


def main() -> None:
    load_dotenv()
    graph = build_graph()
    thread_id = get_env("THREAD_ID", "arag-cli")
    show_sources = get_env("SHOW_SOURCES", "false").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    print("ARAG terminal agent ready. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            user_input = input("You> ").strip()
        except EOFError:
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        result = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        assistant_message = result["messages"][-1]
        print(f"Assistant> {assistant_message.content}")
        if show_sources:
            print_sources(result.get("context", []))


if __name__ == "__main__":
    main()
