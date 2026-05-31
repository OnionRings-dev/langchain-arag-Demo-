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
from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

console = Console()


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


def display_documents(docs: list[Document]) -> None:
    if not docs:
        console.print("[yellow]No relevant documents found.[/yellow]")
        return

    table = Table(
        title="Retrieved Documents",
        show_header=True,
        header_style="bold magenta",
        expand=True,
    )
    table.add_column("#", style="dim", width=2)
    table.add_column("Source", style="cyan", no_wrap=False, overflow="fold")
    table.add_column("Content Snippet", style="white")

    for i, doc in enumerate(docs, start=1):
        source_val = (
            doc.metadata.get("source")
            or doc.metadata.get("url")
            or doc.metadata.get("file")
            or "Unknown"
        )
        # Create a clickable link if it's a URL
        if str(source_val).startswith("http"):
            source_display = f"[link={source_val}]{source_val}[/link]"
        else:
            source_display = str(source_val)

        content = doc.page_content.replace("\n", " ")[:150] + "..."
        table.add_row(str(i), source_display, content)

    console.print(table)


def main() -> None:
    load_dotenv()

    with console.status("[bold green]Building graph...", spinner="dots"):
        graph = build_graph()

    thread_id = get_env("THREAD_ID", "arag-cli")

    console.print(
        Panel.fit(
            "[bold blue]ARAG Terminal Agent[/bold blue]\n[dim]Type 'exit' or 'quit' to stop.[/dim]",
            border_style="blue",
        )
    )

    while True:
        try:
            user_input = console.input("[bold green]You>[/bold green] ").strip()
        except EOFError:
            console.print()
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        # Use streaming to show "tool calls" (node executions)
        with console.status(
            "[bold yellow]Processing...", spinner="bouncingBar"
        ) as status:
            last_state = {}
            for update in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                for node_name, state_update in update.items():
                    if node_name == "retrieve":
                        status.update(
                            f"[bold cyan]Retrieved {len(state_update.get('context', []))} documents..."
                        )
                        last_state.update(state_update)
                        console.print(f"\n[bold cyan]Step: {node_name}[/bold cyan]")
                        display_documents(state_update.get("context", []))
                    elif node_name == "generate":
                        status.update("[bold magenta]Generating response...")
                        last_state.update(state_update)
                        console.print(f"[bold magenta]Step: {node_name}[/bold magenta]")

            # After streaming is done, print the final message
            if "messages" in last_state:
                assistant_message = last_state["messages"][-1]
                console.print(
                    Panel(
                        Markdown(assistant_message.content),
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                )


if __name__ == "__main__":
    main()
