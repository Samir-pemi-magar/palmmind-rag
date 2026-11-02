# Palm Mind RAG API

Document ingestion and conversational RAG backend for Palm Mind AI, built with FastAPI.

Upload PDF or TXT documents, ask questions about them in a multi-turn chat, and book interview slots through the same chat endpoint.

## Features

- **Document ingestion**: upload one or more PDF or TXT files (up to 20 MB each) in a single request, extract text, chunk it, embed it, and store it for search. A bad file does not stop the others.
- **Document management**: list uploaded documents, look one up by ID, or delete it. Deleting removes its vectors from Qdrant, its database record, and the stored file.
- **Two chunking strategies**: `fixed` (character windows with overlap) and `recursive` (splits on paragraph, line, and sentence boundaries where possible).
- **Conversational RAG**: follow-up questions are rewritten into standalone search queries, relevant chunks are retrieved from Qdrant, and a local LLM answers using only that context. Every answer includes its sources.
- **Conversation memory**: the last 10 messages of each conversation are kept in Redis for 24 hours.
- **Interview booking**: a multi-step booking flow inside the chat. It collects name, email, date, time, and timezone, asks for explicit confirmation, then saves the request to the database.

## Tech stack

| Concern | Technology |
| --- | --- |
| API | FastAPI, Uvicorn |
| Vector store | Qdrant (cosine similarity) |
| Embeddings | BGE-M3 via FlagEmbedding (1024 dimensions, runs on CPU) |
| LLM | Any OpenAI-compatible server (default model: Qwen3-8B Q4_K_M) |
| Conversation and booking state | Redis |
| Relational data | SQLite via SQLAlchemy (async) and Alembic |
| PDF extraction | PyMuPDF |
| Tooling | uv, Ruff, pytest |

## How it works

### Ingestion

```
Upload -> Save file -> Extract text -> Chunk -> Embed (BGE-M3) -> Qdrant -> SQLite metadata
```

If any step fails, the saved file and any already-written Qdrant points are cleaned up.

### Chat

```
Message
  |
  +--> Booking handler (LLM extracts intent and details, Redis holds state)
  |        |
  |        +-- booking-related --> reply with the next question, summary, or confirmation
  |        |
  |        +-- not booking-related
  |                 |
  +-----------------+--> Rewrite follow-up as a standalone query
                         -> Embed -> Qdrant search -> Build context -> LLM answer + sources
```

Booking messages are checked first. Anything that is not booking-related falls through to the RAG pipeline. Both kinds of exchange are saved to conversation history.

## Project structure

```
app/
  main.py                  FastAPI app and lifespan (creates the Qdrant collection)
  api/routes/              HTTP endpoints (documents, chat, health)
  core/                    Config, database, Qdrant, Redis, embeddings, LLM client
  models/                  SQLAlchemy models (Document, InterviewBooking)
  schemas/                 Pydantic request and response models
  repositories/            Database access (bookings)
  services/
    ingestion.py           Upload-to-index pipeline
    document_management.py List, fetch, and delete documents
    document_storage.py    File validation and saving
    document_extractor.py  PDF and TXT text extraction
    chunking.py            Fixed and recursive chunking
    retrieval.py           Vector search with optional document filter
    rag.py                 Query rewriting, prompt building, answer generation
    chat.py                Chat orchestration (booking first, then RAG)
    conversation_memory.py Redis chat history
    booking*.py            Booking flow, extraction, validation, Redis state
migrations/                Alembic migrations
tests/                     Test suite
compose.yaml               Qdrant and Redis
Makefile                   Common development commands
```

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Docker with Docker Compose
- The BGE-M3 embedding model downloaded locally
- An OpenAI-compatible LLM server serving a chat model (the project is configured for Qwen3-8B)

## Getting started

### 1. Install dependencies

```bash
make install
```

### 2. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and set at least `EMBEDDING_MODEL_PATH` to the folder containing BGE-M3. Check that `LLM_BASE_URL` and `LLM_MODEL` match your LLM server.

### 3. Start Qdrant and Redis

```bash
make up
```

### 4. Apply database migrations

```bash
make migrate
```

### 5. Run the API

```bash
make api
```

The API is served at `http://127.0.0.1:8000`, with interactive docs at `http://127.0.0.1:8000/docs`.

### Shortcut: `make dev`

`make dev` does steps 3 to 5 in one command. It starts Qdrant and Redis, starts an existing Docker container named `palmmind-llm`, waits for the LLM server on port 3000, applies migrations, starts the API in the background, and opens the docs page. Stop everything with `make dev-stop`.

This shortcut is tied to the author's setup: it expects the `palmmind-llm` container to already exist and uses `cmd.exe` to open the browser, so it works as-is on WSL only. On other systems, use the manual steps above.

## Configuration

All settings are read from environment variables or `.env`.

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | `Palm Mind RAG API` | Application name |
| `APP_VERSION` | `0.1.0` | Application version |
| `DEBUG` | `true` | Enables SQL query logging |
| `DATABASE_URL` | `sqlite+aiosqlite:///./palmmind.db` | Database connection string |
| `UPLOAD_DIR` | `./data/uploads` | Where uploaded files are stored |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant address |
| `QDRANT_COLLECTION` | `document_chunks` | Qdrant collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis address |
| `LLM_BASE_URL` | `http://localhost:3000/v1` | Base URL of the OpenAI-compatible LLM server |
| `LLM_MODEL` | `Qwen3-8B-Q4_K_M.gguf` | Model name sent with each request |
| `LLM_API_KEY` | `not-needed` | API key, if your server requires one |
| `EMBEDDING_MODEL_PATH` | `./models/bge-m3` | Local path to BGE-M3 |
| `EMBEDDING_DIMENSION` | `1024` | Must match the embedding model |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `75` | Characters shared between neighbouring chunks |
| `RETRIEVAL_TOP_K` | `5` | Chunks retrieved per question |

## API reference

### `GET /health`

Returns the application status.

### `POST /documents`

Upload and index one or more documents. Sent as `multipart/form-data`.

| Field | Type | Description |
| --- | --- | --- |
| `uploads` | file (repeatable) | One or more `.pdf` or `.txt` files, up to 20 MB each |
| `strategy` | string | `fixed` or `recursive`, applied to every file in the request |

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -F "uploads=@handbook.pdf" \
  -F "uploads=@faq.txt" \
  -F "strategy=recursive"
```

Response (`200`):

```json
{
  "total_files": 2,
  "successful": [
    {
      "id": "0b6f6c9e-6c3a-4a1e-9d57-2f3f6f0a1c11",
      "filename": "handbook.pdf",
      "status": "completed",
      "chunking_strategy": "recursive",
      "total_chunks": 42,
      "created_at": "2026-11-02T10:15:00Z"
    }
  ],
  "failed": [
    {
      "filename": "faq.txt",
      "error": "Uploaded file is empty."
    }
  ]
}
```

Files are processed one at a time. A file that fails (unsupported type, empty, too large, no extractable text) is reported in `failed` and the remaining files are still ingested, so the request returns `200` even when some files fail. A missing `uploads` field or an invalid `strategy` returns `422`.

### `GET /documents`

Returns all uploaded documents, newest first, as a list of the same document objects shown above.

### `GET /documents/{document_id}`

Returns one document. Responds with `404` if the ID does not exist.

### `DELETE /documents/{document_id}`

Deletes a document: its vectors in Qdrant, its database record, and the original file in the upload directory. Responds with `404` if the ID does not exist.

```json
{
  "message": "Document deleted successfully.",
  "document_id": "0b6f6c9e-6c3a-4a1e-9d57-2f3f6f0a1c11",
  "filename": "handbook.pdf"
}
```

Vectors are removed first, so if Qdrant is unavailable the database record and file are left untouched and the delete can be retried. The file is only removed if it sits inside the upload directory.

### `POST /chat`

Send a message. Omit `conversation_id` to start a new conversation, and reuse the returned one to continue it.

| Field | Type | Description |
| --- | --- | --- |
| `message` | string | Required, 1 to 10,000 characters |
| `conversation_id` | string | Optional, continues an existing conversation |
| `document_id` | string | Optional, restricts retrieval to one document |

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the leave policy?"}'
```

Response:

```json
{
  "conversation_id": "5c1d2f8e-1d0b-4f6a-8d4c-7a9e6b3c2f10",
  "answer": "Employees are entitled to ...",
  "sources": [
    {
      "document_id": "0b6f6c9e-6c3a-4a1e-9d57-2f3f6f0a1c11",
      "filename": "handbook.pdf",
      "chunk_index": 7,
      "score": 0.83
    }
  ]
}
```

If nothing relevant is found, the assistant says so and `sources` is empty.

## Interview booking flow

Booking happens through `/chat` using the same `conversation_id` across messages:

1. The candidate says they want to book an interview.
2. The assistant asks for any missing detail, one at a time: full name, email, date, time, and an IANA timezone such as `Asia/Kathmandu`.
3. Each detail is validated as it arrives (email format, known timezone, date must be in the future). Details can be corrected at any point.
4. Once everything is collected, the assistant shows a summary and asks for confirmation.
5. Replying `yes` saves the booking with status `pending` and returns a reference ID. Replying `no` or `cancel` discards it.

In-progress booking details live in Redis for 24 hours and are only written to the database after explicit confirmation. Saved bookings are requests, not confirmed appointments.

## Development

| Command | Description |
| --- | --- |
| `make help` | List all commands |
| `make test` | Run the test suite |
| `make test-verbose` | Run tests with verbose output |
| `make lint` | Check code with Ruff |
| `make format` | Format code with Ruff |
| `make migrate` | Apply database migrations |
| `make migration-check` | Verify models and migrations are in sync |
| `make migration-history` | Show migration history |
| `make check` | Run lint, migration check, and tests |
| `make logs` | Follow Qdrant and Redis logs |
| `make down` | Stop Qdrant and Redis (data is kept) |
| `make dev` | Start everything and open the API docs (WSL, see above) |
| `make dev-stop` | Stop the API, Qdrant, and Redis |

To create a new migration after changing a model:

```bash
uv run alembic revision --autogenerate -m "describe the change"
```

## Notes

- Only PDF and TXT files are supported. Scanned PDFs without a text layer will be rejected.
- Embeddings run on CPU. The embedding model and PDF extractor are loaded on first use, so the API starts quickly but the first upload or chat after startup is slower.
- Chat and retrieval can be limited to one document by passing its `document_id`. Deleting a document removes it from future answers immediately.
- Retrieved document text is treated as untrusted data, and the assistant is instructed not to follow instructions found inside documents.
