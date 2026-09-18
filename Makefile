.PHONY: help setup test run probe clean

help:
	@echo "make setup   install dependencies into .venv"
	@echo "make test    run the offline policy tests (no key, no network)"
	@echo "make run     triage the sample tickets against Jev (needs a key in .env)"
	@echo "make probe   discover how OpenRouter serves Jev (needs a key in .env)"
	@echo "make clean   remove build and cache artifacts"

setup:
	uv sync

test:
	uv run pytest

run:
	uv run examples/01_ticket_triage.py

probe:
	uv run scripts/probe_openrouter.py

clean:
	$(RM) -r .pytest_cache .ruff_cache dist build
	find . -name __pycache__ -type d -prune -exec $(RM) -r {} +
