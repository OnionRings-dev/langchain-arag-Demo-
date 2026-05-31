## ARAG terminal agent

This project runs a LangChain + LangGraph terminal agent that queries your Qdrant
collection and uses Groq for chat inference. It is designed to test the quality
of your scraping, chunking, and embedding pipeline in a conversational loop.

### Features

- **Advanced Terminal UI**: Uses the `rich` library for an interactive experience with spinners, formatted tables, and Markdown support.
- **Reranking**: Implements a two-stage retrieval process using `flashrank` (`ms-marco-MiniLM-L-12-v2`) to improve context relevance.
- **Graph-based Workflow**: Built with `LangGraph` for clear separation between retrieval, reranking, and generation steps.

### Setup

1. Create a `.env` file with your Groq key and Qdrant settings:

```
GROQ_API_KEY=your_groq_key
GROQ_MODEL=openai/gpt-oss-20b
GROQ_TEMPERATURE=0.2

# Qdrant (your live Docker instance)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=arag
QDRANT_DOMAIN=example.com
QDRANT_API_KEY=

# Payload keys used by your ingestion pipeline
QDRANT_TEXT_KEY=text
QDRANT_METADATA_KEY=metadata

# Optional Filtering
QDRANT_FILTER_KEY=document_type
QDRANT_FILTER_VALUE=

# Embeddings
EMBEDDING_PROVIDER=fastembed
FASTEMBED_MODEL=BAAI/bge-small-en-v1.5

# Retrieval
TOP_K=4
SHOW_SOURCES=false
THREAD_ID=arag-cli
```

If you used LangChain's default Qdrant ingestion, set `QDRANT_TEXT_KEY=page_content`.

2. Install dependencies:

```
uv sync
```

### Run

```
uv run python main.py
```

Type `exit` or `quit` to stop the chat.

### Scripts

Utility scripts live in `scripts/` and read `QDRANT_URL`/`QDRANT_API_KEY` from `.env`.
For example: `uv run python scripts/qdrant_collections_rows.py` and
`uv run python scripts/qdrant_collection_info.py <collection>`.
