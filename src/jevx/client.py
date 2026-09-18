"""Pick a provider for Jev and hand back a client with one `ask` method.

Jev is reachable two ways, and both speak the same System One protocol -- state
plus named questions in, typed answers with probabilities out -- so the examples
do not care which one answers:

    TYPESAFE_API_KEY set     ->  api.typesafe.ai, via typesafe_sdk (early access)
    OPENROUTER_API_KEY set   ->  OpenRouter, model typesafe/jev-1.13

The two need different transports. OpenRouter serves decision models on a
dedicated endpoint, `POST /api/alpha/decisions`, rather than through
`/chat/completions`:

    typesafe/jev-1.13 is a decisions model and cannot be used with the
    chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.

The SDK builds its own `/v1/systemone` path, so it cannot be repointed there
with `base_url` alone. `OpenRouterClient` posts to the decisions endpoint
directly and parses the reply with the SDK's own `SystemOneResponse`, which
means both paths return the identical typed object.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dotenv import load_dotenv
from typesafe_sdk import SystemOneResponse, TypeSafeClient

REPO_ROOT = Path(__file__).resolve().parents[2]

#: OpenRouter's endpoint for decision models. Verified against the live service.
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

#: Jev on OpenRouter. `typesafe/jev-1.13-20260917` is the pinned build.
OPENROUTER_MODEL_DEFAULT = "typesafe/jev-1.13"

#: Jev on TypeSafe directly.
TYPESAFE_MODEL_DEFAULT = "jev-latest"

DEFAULT_TIMEOUT_SECONDS = 60


class MissingKeyError(RuntimeError):
    """Raised when no usable credentials were found."""


class ProviderError(RuntimeError):
    """Raised when a provider rejects a request."""


@dataclass(frozen=True)
class Provider:
    """Which service answers, and as which model."""

    name: str
    model: str
    api_key: str
    endpoint: str

    @property
    def label(self) -> str:
        """Provider and model, safe to print: never includes the key."""
        return f"{self.name} · {self.model}"


class JevClient(Protocol):
    """The one operation the examples need from a provider."""

    def ask(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        """Answer every question against the state in a single request."""

    def close(self) -> None:
        """Release any underlying resources."""


def _serialise_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Convert SDK question objects to their JSON form.

    Args:
        questions: Question objects keyed by name.

    Returns:
        The same mapping with each question dumped to plain JSON types.
    """
    return {
        name: question.model_dump(exclude_none=True) if hasattr(question, "model_dump") else question
        for name, question in questions.items()
    }


class OpenRouterClient:
    """Call Jev through OpenRouter's decisions endpoint."""

    def __init__(self, provider: Provider, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        """Hold the provider configuration for subsequent requests."""
        self._provider = provider
        self._timeout = timeout

    def ask(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        """Answer every question against the state in a single request.

        Args:
            state: The content to judge. A string, or JSON-shaped object.
            questions: SDK question objects keyed by name.

        Returns:
            The parsed response, identical in type to the native SDK's.

        Raises:
            ProviderError: When OpenRouter returns a non-2xx response.
        """
        payload = {
            "model": self._provider.model,
            "state": state,
            "questions": _serialise_questions(questions),
        }
        request = urllib.request.Request(
            self._provider.endpoint,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._provider.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rajivkuriakose/typesafe-jev-examples",
                "X-Title": "typesafe-jev-examples",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                body = response.read().decode()
        except urllib.error.HTTPError as error:
            detail = error.read().decode()[:500]
            raise ProviderError(f"OpenRouter returned {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ProviderError(f"Could not reach OpenRouter: {error.reason}") from error

        return SystemOneResponse.model_validate_json(body)

    def close(self) -> None:
        """Nothing to release: each request opens its own connection."""

    def __enter__(self) -> OpenRouterClient:
        """Enter a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Leave a context manager."""
        self.close()


class TypeSafeNativeClient:
    """Call Jev through TypeSafe's own API, using the official SDK."""

    def __init__(self, provider: Provider) -> None:
        """Build an SDK client for the provider."""
        self._client = TypeSafeClient(api_key=provider.api_key, model=provider.model)

    def ask(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        """Answer every question against the state in a single request.

        Args:
            state: The content to judge.
            questions: SDK question objects keyed by name.

        Returns:
            The parsed response.
        """
        return self._client.system_one(state=state, questions=questions)

    def close(self) -> None:
        """Close the underlying SDK client."""
        self._client.close()

    def __enter__(self) -> TypeSafeNativeClient:
        """Enter a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Leave a context manager."""
        self.close()


def resolve_provider(env: dict[str, str] | None = None) -> Provider:
    """Choose a provider from the environment.

    TypeSafe direct wins when its key is present, since an early-access key is
    the more authoritative path. Otherwise OpenRouter.

    Args:
        env: Environment mapping to read. Defaults to `os.environ`.

    Returns:
        The resolved provider.

    Raises:
        MissingKeyError: When neither key is set.
    """
    env = dict(os.environ if env is None else env)

    typesafe_key = (env.get("TYPESAFE_API_KEY") or "").strip()
    if typesafe_key:
        return Provider(
            name="typesafe",
            model=(env.get("JEV_MODEL") or "").strip() or TYPESAFE_MODEL_DEFAULT,
            api_key=typesafe_key,
            endpoint="https://api.typesafe.ai/v1/systemone",
        )

    openrouter_key = (env.get("OPENROUTER_API_KEY") or "").strip()
    if openrouter_key:
        return Provider(
            name="openrouter",
            model=(env.get("JEV_MODEL") or "").strip() or OPENROUTER_MODEL_DEFAULT,
            api_key=openrouter_key,
            endpoint=(env.get("JEV_DECISIONS_URL") or "").strip() or OPENROUTER_DECISIONS_URL,
        )

    raise MissingKeyError(
        "No API key found. Copy .env.example to .env and set OPENROUTER_API_KEY "
        "(https://openrouter.ai/settings/keys), or TYPESAFE_API_KEY if your "
        "early-access key has arrived."
    )


def open_client(env: dict[str, str] | None = None) -> tuple[JevClient, Provider]:
    """Load `.env`, resolve a provider, and build a client for it.

    Args:
        env: Environment mapping to read. Defaults to the process environment
            after `.env` has been loaded.

    Returns:
        A client and the provider it was configured for. Use the client as a
        context manager, or close it when done.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    provider = resolve_provider(env)
    client: JevClient = (
        TypeSafeNativeClient(provider) if provider.name == "typesafe" else OpenRouterClient(provider)
    )
    return client, provider
