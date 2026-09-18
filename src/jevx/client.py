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
means both providers return the identical typed object.

One difference remains. The SDK's internal decoder drops answer types it does
not recognise before validating, so a future primitive would be ignored rather
than fatal. Parsing the body directly, as the OpenRouter path must, keeps
pydantic's strictness: an unknown answer type raises instead. That is the right
trade for an examples repository -- surfacing the surprise beats hiding it --
but it does make this path marginally less forgiving than the native one.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol

import pydantic
from dotenv import load_dotenv
from typesafe_sdk import SystemOneResponse, TypeSafeClient

REPO_ROOT = Path(__file__).resolve().parents[2]

#: OpenRouter's endpoint for decision models. Verified against the live service.
OPENROUTER_DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

#: Jev on OpenRouter. `typesafe/jev-1.13-20260917` is the pinned build.
OPENROUTER_MODEL_DEFAULT = "typesafe/jev-1.13"

#: Jev on TypeSafe directly. Matches typesafe_sdk's own DEFAULT_MODEL.
TYPESAFE_MODEL_DEFAULT = "jev-latest"

DEFAULT_TIMEOUT_SECONDS = 60

#: Statuses worth sending the same request again for: rate limiting, and the
#: gateway and origin errors a hosted service emits transiently. A 4xx other
#: than 429 means the request itself is wrong, so retrying only costs money.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504, 520, 521, 522, 524})

#: Total attempts, not retries after the first.
MAX_ATTEMPTS = 3

#: First backoff, doubled each attempt.
BACKOFF_SECONDS = 0.5


class MissingKeyError(RuntimeError):
    """Raised when no usable credentials were found."""


class ProviderError(RuntimeError):
    """Raised when a provider rejects a request or answers unintelligibly."""


@dataclass(frozen=True)
class Provider:
    """Which service answers, and as which model."""

    name: str
    model: str
    api_key: str
    #: Where to POST. `None` for the native path, where the SDK builds its own
    #: URL and no endpoint of ours is consulted.
    endpoint: str | None = None

    @property
    def label(self) -> str:
        """Provider and model, safe to print: never includes the key."""
        return f"{self.name} · {self.model}"


class JevClient(Protocol):
    """The operations the examples need from a provider."""

    def ask(self, state: Any, questions: dict[str, Any]) -> SystemOneResponse:
        """Answer every question against the state in a single request."""

    def close(self) -> None:
        """Release any underlying resources."""

    def __enter__(self) -> JevClient:
        """Enter a context manager."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a context manager, releasing resources."""


def _serialise_questions(questions: dict[str, Any]) -> dict[str, Any]:
    """Convert SDK question objects to their JSON form.

    The dump must retain each question's `type` discriminator -- `choice`,
    `noul` or `score` -- because that is what tells the service which primitive
    to answer with. A test asserts this, since a silent change here would fail
    only at runtime, against the live API.

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
            ProviderError: When OpenRouter is unreachable, returns a non-2xx
                response, or answers with a body that is not a System One
                response.
        """
        payload = {
            "model": self._provider.model,
            "state": state,
            "questions": _serialise_questions(questions),
        }
        request = urllib.request.Request(
            self._provider.endpoint or OPENROUTER_DECISIONS_URL,
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self._provider.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/rajivkuriakose/typesafe-jev-examples",
                "X-Title": "typesafe-jev-examples",
            },
            method="POST",
        )
        body = self._read_with_retries(request)

        # A 2xx with an unexpected shape is still a provider failure, not a bug
        # in the caller, so it leaves here as ProviderError like everything else.
        try:
            return SystemOneResponse.model_validate_json(body)
        except pydantic.ValidationError as error:
            raise ProviderError(
                f"OpenRouter returned a body that is not a System One response: {body[:300]}"
            ) from error

    def _read_with_retries(self, request: urllib.request.Request) -> str:
        """Send the request, retrying the failures that are worth retrying.

        A long fan-out makes rare transient errors near-certain: at 156 calls a
        one-in-two-hundred gateway blip is likely to land, and without this the
        whole run dies on it. Retries are finite and only for statuses that a
        second identical request could plausibly answer.

        Args:
            request: The prepared POST.

        Returns:
            The response body.

        Raises:
            ProviderError: When the request fails and is not worth retrying, or
                when the attempts are exhausted.
        """
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.read().decode()
            except urllib.error.HTTPError as error:
                retryable = error.code in RETRY_STATUSES
                if not retryable or attempt == MAX_ATTEMPTS:
                    detail = error.read().decode()[:500]
                    raise ProviderError(f"OpenRouter returned {error.code}: {detail}") from error
            except urllib.error.URLError as error:
                if attempt == MAX_ATTEMPTS:
                    raise ProviderError(f"Could not reach OpenRouter: {error.reason}") from error
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))

        raise ProviderError("OpenRouter could not be reached")  # pragma: no cover

    def close(self) -> None:
        """Nothing to release: each request opens its own connection."""

    def __enter__(self) -> OpenRouterClient:
        """Enter a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a context manager."""
        self.close()


class TypeSafeNativeClient:
    """Call Jev through TypeSafe's own API, using the official SDK.

    Untested against the live service: it needs an early-access key, which this
    repository's author does not yet have. The SDK surface it uses is verified
    against typesafe_sdk 0.7.0, but treat the path itself as unproven.
    """

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

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Leave a context manager."""
        self.close()


def resolve_provider(env: dict[str, str] | None = None) -> Provider:
    """Choose a provider from the environment.

    TypeSafe direct wins when its key is present, since an early-access key is
    the more authoritative path. Otherwise OpenRouter.

    The two providers name their models in different namespaces -- `jev-latest`
    against TypeSafe, `typesafe/jev-1.13` against OpenRouter -- so each reads
    its own override variable. Sharing one would send an OpenRouter model id to
    api.typesafe.ai, which rejects it.

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
            model=(env.get("TYPESAFE_MODEL") or "").strip() or TYPESAFE_MODEL_DEFAULT,
            api_key=typesafe_key,
            endpoint=None,
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
        env: Environment mapping to read. When given, `.env` is left alone and
            the process environment is not touched, which keeps tests isolated.

    Returns:
        A client and the provider it was configured for. Use the client as a
        context manager, or close it when done.
    """
    if env is None:
        load_dotenv(REPO_ROOT / ".env", override=False)
    provider = resolve_provider(env)
    client: JevClient = (
        TypeSafeNativeClient(provider) if provider.name == "typesafe" else OpenRouterClient(provider)
    )
    return client, provider
