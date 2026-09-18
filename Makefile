.PHONY: help setup test test-live run rerank probe clean

help:
	@echo "make setup   install dependencies into .venv"
	@echo "make test    run the offline tests (no key, no network)"
	@echo "make test-live  also run tests that call Jev (needs a key in .env)"
	@echo "make run     triage the sample tickets against Jev (needs a key in .env)"
	@echo "make rerank  re-rank help-center search results (needs a key in .env)"
	@echo "make probe   discover how OpenRouter serves Jev (needs a key in .env)"
	@echo "make clean   remove build and cache artifacts"

setup:
	uv sync

test:
	uv run pytest

test-live:
	uv run pytest -m live

run:
	uv run examples/01_ticket_triage.py

rerank:
	uv run examples/02_rerank.py

probe:
	uv run scripts/probe_openrouter.py

clean:
	$(RM) -r .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -exec $(RM) -r {} +
