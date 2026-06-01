from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal, TypedDict

from dotenv import load_dotenv
from flashrank import Ranker, RerankRequest
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from qdrant_client import QdrantClient, models
from rich.console import Console
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
    top_k = int(get_env("TOP_K", "20"))

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

    # Initialize Ranker once
    ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank")

    @tool
    def query_knowledge_base(query: str) -> str:
        """Consult the knowledge base to get MORE information.
        Use this ONLY if the initial context provided is not enough to answer the user's question.
        """
        # 1. Retrieve
        docs = retriever.invoke(query)
        if not docs:
            return "No additional relevant documents found."

        # 2. Rerank
        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)

        top_results = results[:5]
        formatted = []
        for i, r in enumerate(top_results, start=1):
            source = r["meta"].get("source") or r["meta"].get("url") or "Unknown"
            formatted.append(f"Source [{i}]: {source}\nContent: {r['text']}")

        return "\n\n".join(formatted)

    tools = [query_knowledge_base]
    llm_with_tools = llm.bind_tools(tools)

    def retrieve(state: GraphState):
        question = latest_user_message(state["messages"])
        docs = retriever.invoke(question)
        return {"context": docs}

    def rerank(state: GraphState):
        question = latest_user_message(state["messages"])
        docs = state.get("context", [])
        if not docs:
            return {"context": []}

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        results = ranker.rerank(RerankRequest(query=question, passages=passages))

        reranked_docs = [
            Document(page_content=r["text"], metadata=r["meta"]) for r in results[:5]
        ]

        # Inject initial context as a system message for the agent
        initial_context = format_docs(reranked_docs)
        context_msg = (
            "system",
            f"Initial Context from Knowledge Base:\n\n{initial_context}",
        )

        return {"context": reranked_docs, "messages": [context_msg]}

    def agent(state: GraphState):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an ARAG terminal assistant. You have been provided with initial context. "
                    "If the initial context is sufficient, answer directly. "
                    "If you need more specific details, use the 'query_knowledge_base' tool. "
                    "Always cite your sources using [Index] or the Source URL.",
                ),
                MessagesPlaceholder("messages"),
            ]
        )
        chain = prompt | llm_with_tools
        msg = chain.invoke(state["messages"])
        return {"messages": [msg]}

    def should_continue(state: GraphState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        return "tools" if last_message.tool_calls else END

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("rerank", rerank)
    workflow.add_node("agent", agent)
    workflow.add_node("tools", ToolNode(tools))

    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())


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

        # Use streaming to show "thoughts" and tool calls
        with console.status(
            "[bold yellow]Agent is thinking...", spinner="bouncingBar"
        ) as status:
            for update in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                for node_name, state_update in update.items():
                    if node_name == "rerank":
                        # Initial RAG results
                        console.print(f"\n[bold cyan]Step: Traditional RAG[/bold cyan]")
                        display_documents(state_update.get("context", []))
                    elif node_name == "agent":
                        msg = state_update["messages"][-1]
                        if msg.tool_calls:
                            for tc in msg.tool_calls:
                                console.print(
                                    Panel(
                                        f"[bold cyan]Tool:[/bold cyan] {tc['name']}\n[bold cyan]Args:[/bold cyan] {json.dumps(tc['args'], indent=2)}",
                                        title="[bold yellow]Autonomous Expansion Search[/bold yellow]",
                                        border_style="yellow",
                                    )
                                )
                        elif msg.content:
                            # This is the final answer or a partial thought
                            pass
                    elif node_name == "tools":
                        # Tool execution finished
                        status.update(
                            "[bold green]Analyzing additional search results..."
                        )

            # Final state retrieval to show the answer
            final_state = graph.get_state(
                config={"configurable": {"thread_id": thread_id}}
            )
            last_message = final_state.values["messages"][-1]
            if last_message.content:
                console.print(
                    Panel(
                        Markdown(last_message.content),
                        title="[bold blue]Assistant[/bold blue]",
                        border_style="blue",
                        expand=False,
                    )
                )


if __name__ == "__main__":
    main()
