"""Provider resolution and question serialisation are pure, so test them directly.

`resolve_provider` decides which service gets your key, and
`_serialise_questions` decides what goes on the wire. Both are cheap to get
subtly wrong and expensive to debug against a live API, so both are pinned here
with no key and no network.
"""

from __future__ import annotations

import io

import pytest
from typesafe_sdk import Choice, Noul, Score

from jevx.client import (
    OPENROUTER_DECISIONS_URL,
    ProviderError,
    OPENROUTER_MODEL_DEFAULT,
    TYPESAFE_MODEL_DEFAULT,
    MissingKeyError,
    Provider,
    _serialise_questions,
    resolve_provider,
)


def test_an_openrouter_key_selects_openrouter():
    provider = resolve_provider({"OPENROUTER_API_KEY": "or-test"})
    assert provider.name == "openrouter"
    assert provider.model == OPENROUTER_MODEL_DEFAULT
    assert provider.endpoint == OPENROUTER_DECISIONS_URL


def test_a_typesafe_key_selects_typesafe():
    provider = resolve_provider({"TYPESAFE_API_KEY": "ts-test"})
    assert provider.name == "typesafe"
    assert provider.model == TYPESAFE_MODEL_DEFAULT
    assert provider.endpoint is None, "the SDK builds its own URL on this path"


def test_typesafe_wins_when_both_keys_are_present():
    provider = resolve_provider({"TYPESAFE_API_KEY": "ts", "OPENROUTER_API_KEY": "or"})
    assert provider.name == "typesafe"


def test_the_two_providers_do_not_share_a_model_override():
    """JEV_MODEL names an OpenRouter model; api.typesafe.ai would reject it.

    .env.example ships JEV_MODEL set. Someone who later adds TYPESAFE_API_KEY
    must not silently start sending `typesafe/jev-1.13` to the direct API.
    """
    provider = resolve_provider(
        {"TYPESAFE_API_KEY": "ts", "JEV_MODEL": "typesafe/jev-1.13"}
    )
    assert provider.model == TYPESAFE_MODEL_DEFAULT

    provider = resolve_provider({"TYPESAFE_API_KEY": "ts", "TYPESAFE_MODEL": "jev-preview"})
    assert provider.model == "jev-preview"


def test_openrouter_overrides_are_honoured():
    provider = resolve_provider(
        {
            "OPENROUTER_API_KEY": "or",
            "JEV_MODEL": "typesafe/jev-1.13-20260917",
            "JEV_DECISIONS_URL": "https://example.invalid/decisions",
        }
    )
    assert provider.model == "typesafe/jev-1.13-20260917"
    assert provider.endpoint == "https://example.invalid/decisions"


@pytest.mark.parametrize("env", [{}, {"OPENROUTER_API_KEY": "   "}, {"TYPESAFE_API_KEY": ""}])
def test_a_blank_or_absent_key_is_no_key(env):
    with pytest.raises(MissingKeyError):
        resolve_provider(env)


def test_the_label_never_exposes_the_key():
    provider = Provider(name="openrouter", model="typesafe/jev-1.13", api_key="or-secret-value")
    assert "or-secret-value" not in provider.label
    assert provider.label == "openrouter · typesafe/jev-1.13"


def test_serialised_questions_keep_their_type_discriminator():
    """The `type` field is what tells the service which primitive to answer with.

    If a future SDK version stopped emitting it, every request would fail at
    runtime against the live API and nothing offline would catch it.
    """
    serialised = _serialise_questions(
        {
            "a": Choice(instructions="Pick one.", criteria={"x": "X.", "y": "Y."}),
            "b": Noul(instructions="Is it so?"),
            "c": Score(instructions="How much?", criteria=["Low.", "High."]),
        }
    )
    assert serialised["a"]["type"] == "choice"
    assert serialised["b"]["type"] == "noul"
    assert serialised["c"]["type"] == "score"
    assert serialised["a"]["criteria"] == {"x": "X.", "y": "Y."}
    assert serialised["c"]["criteria"] == ["Low.", "High."]


def test_serialisation_drops_unset_fields_rather_than_sending_nulls():
    serialised = _serialise_questions({"b": Noul(instructions="Is it so?")})
    assert None not in serialised["b"].values()


# --- transient failure handling -------------------------------------------


def _http_error(code):
    import urllib.error

    return urllib.error.HTTPError(
        url="https://openrouter.ai/api/alpha/decisions",
        code=code,
        msg="boom",
        hdrs=None,
        fp=io.BytesIO(b'{"error":{"message":"boom"}}'),
    )


class _Body(io.BytesIO):
    """A urlopen context-manager stand-in."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


GOOD_BODY = (
    b'{"model":"typesafe/jev-1.13","answers":{"relevant":{"type":"noul","noul":0.8}},'
    b'"usage":{"input_tokens":10,"output_tokens":2}}'
)


def test_a_transient_upstream_error_is_retried(monkeypatch):
    """A single flaky 520 must not kill a 156-call run."""
    from jevx import client as client_module

    attempts = []

    def flaky(request, timeout=None):
        attempts.append(1)
        if len(attempts) < 3:
            raise _http_error(520)
        return _Body(GOOD_BODY)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    provider = Provider(name="openrouter", model="m", api_key="k", endpoint="https://x.invalid")
    response = client_module.OpenRouterClient(provider).ask({"a": 1}, {})

    assert len(attempts) == 3
    assert response.nouls["relevant"].noul == 0.8


def test_a_client_error_is_not_retried(monkeypatch):
    """A 400 means the request is wrong; sending it again just costs money."""
    from jevx import client as client_module

    attempts = []

    def bad_request(request, timeout=None):
        attempts.append(1)
        raise _http_error(400)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", bad_request)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    provider = Provider(name="openrouter", model="m", api_key="k", endpoint="https://x.invalid")
    with pytest.raises(ProviderError, match="400"):
        client_module.OpenRouterClient(provider).ask({"a": 1}, {})

    assert len(attempts) == 1


def test_retries_are_finite(monkeypatch):
    """A persistently broken upstream fails rather than looping."""
    from jevx import client as client_module

    attempts = []

    def always_down(request, timeout=None):
        attempts.append(1)
        raise _http_error(503)

    monkeypatch.setattr(client_module.urllib.request, "urlopen", always_down)
    monkeypatch.setattr(client_module.time, "sleep", lambda seconds: None)

    provider = Provider(name="openrouter", model="m", api_key="k", endpoint="https://x.invalid")
    with pytest.raises(ProviderError, match="503"):
        client_module.OpenRouterClient(provider).ask({"a": 1}, {})

    assert len(attempts) == client_module.MAX_ATTEMPTS
