
# Palm Mind RAG API development commands

.DEFAULT_GOAL := help

.PHONY: help install up down logs api test test-verbose lint format migrate migration-check migration-history check dev dev-stop

help: ## Show available commands
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  %-20s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

install: ## Install project and development dependencies
	uv sync --dev

up: ## Start Qdrant and Redis
	docker compose up -d

down: ## Stop Qdrant and Redis (keeps persisted data)
	docker compose down

logs: ## Follow Qdrant and Redis logs
	docker compose logs -f

api: ## Start the FastAPI server
	uv run uvicorn app.main:app --host 127.0.0.1 --port 8000

test: ## Run the complete test suite
	uv run pytest

test-verbose: ## Run tests with verbose output
	uv run pytest -v

lint: ## Check Python code with Ruff
	uv run ruff check .

format: ## Format Python code with Ruff
	uv run ruff format .

migrate: ## Apply pending database migrations
	uv run alembic upgrade head

migration-check: ## Check whether database models and migrations are synchronized
	uv run alembic check

migration-history: ## Display Alembic migration history
	uv run alembic history

check: lint migration-check test ## Run lint, migration consistency, and tests

dev: up
	@echo "Starting Palm Mind LLM..."
	@if ! docker inspect palmmind-llm >/dev/null 2>&1; then echo "ERROR: palmmind-llm container not found."; exit 1; fi
	@if [ "$$(docker inspect -f '{{.State.Running}}' palmmind-llm)" != "true" ]; then docker start palmmind-llm >/dev/null; fi
	@echo "Waiting for Qwen3-8B..."
	@ready=0; for i in $$(seq 1 90); do if curl -fsS http://127.0.0.1:3000/v1/models >/dev/null 2>&1; then ready=1; break; fi; sleep 2; done; if [ $$ready -ne 1 ]; then echo "ERROR: LLM server did not become ready."; exit 1; fi
	@echo "Applying database migrations..."
	@uv run alembic upgrade head
	@echo "Starting FastAPI..."
	@if ! curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then nohup uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 > /tmp/palmmind-api.log 2>&1 & echo $$! > /tmp/palmmind-api.pid; fi
	@ready=0; for i in $$(seq 1 30); do if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then ready=1; break; fi; sleep 1; done; if [ $$ready -ne 1 ]; then echo "ERROR: FastAPI failed to start. Check /tmp/palmmind-api.log"; exit 1; fi
	@echo "Opening Palm Mind API documentation..."
	@cmd.exe /c start "" "http://127.0.0.1:8000/docs"
	@echo "Palm Mind is ready!"

dev-stop:
	@if [ -f /tmp/palmmind-api.pid ]; then kill $$(cat /tmp/palmmind-api.pid) 2>/dev/null || true; rm -f /tmp/palmmind-api.pid; fi
	@docker compose down
	@echo "Palm Mind development services stopped."
