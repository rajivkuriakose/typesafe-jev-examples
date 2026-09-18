#!/usr/bin/env python3
"""Discover how OpenRouter serves Jev, empirically.

The answer is already known and wired into `src/jevx/client.py`: Jev is a
decisions model, served at `POST /api/alpha/decisions`, taking TypeSafe's own
`state` + `questions` schema. This script re-runs the discovery against the live
service, which is worth doing if OpenRouter moves the endpoint out of alpha.

It sends each candidate shape and reports which answers. Set JEV_DECISIONS_URL in
.env if the working endpoint is no longer the default.

Run:  make probe
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_ROOT = "https://openrouter.ai/api/v1"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"
TIMEOUT_SECONDS = 45

# A deliberately tiny question set: one of each primitive, cheap to send.
STATE = {"ticket": "Payouts have been failing for three days and I'm losing sales."}
QUESTIONS: dict[str, Any] = {
    "is_urgent": {
        "type": "noul",
        "instructions": "The message conveys urgency or time-sensitivity.",
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this ticket.",
        "criteria": {
            "billing": "Charges, refunds, invoices, payouts.",
            "technical": "Bugs, outages, failed integrations.",
        },
    },
}


def _post(url: str, payload: dict[str, Any], key: str) -> tuple[int, str]:
    """POST JSON and return the status and body, without raising on 4xx/5xx.

    Args:
        url: Full request URL.
        payload: JSON-serialisable request body.
        key: Bearer token. Never printed.

    Returns:
        The HTTP status code and the response body as text.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/rajivkuriakose/typesafe-jev-examples",
            "X-Title": "typesafe-jev-examples probe",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read().decode()
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode()
    except urllib.error.URLError as error:
        return 0, f"connection failed: {error.reason}"


def _summarise(body: str, limit: int = 600) -> str:
    """Pretty-print JSON when possible, truncated for terminal output."""
    try:
        return json.dumps(json.loads(body), indent=2)[:limit]
    except json.JSONDecodeError:
        return body[:limit]


def candidates(model: str) -> list[tuple[str, str, dict[str, Any]]]:
    """Return the request shapes worth trying, most likely first.

    Args:
        model: The OpenRouter model id for Jev.

    Returns:
        Tuples of (label, url, payload).
    """
    return [
        (
            "System One protocol at /alpha/decisions  [known good]",
            DECISIONS_URL,
            {"model": model, "state": STATE, "questions": QUESTIONS},
        ),
        (
            "System One protocol, proxied at /v1/systemone",
            f"{OPENROUTER_ROOT}/systemone",
            {"model": model, "state": STATE, "questions": QUESTIONS},
        ),
        (
            "System One payload nested under a chat message",
            f"{OPENROUTER_ROOT}/chat/completions",
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({"state": STATE, "questions": QUESTIONS}),
                    }
                ],
            },
        ),
        (
            "Plain chat completion (does it degrade to text?)",
            f"{OPENROUTER_ROOT}/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": STATE["ticket"]}],
            },
        ),
    ]


def main() -> int:
    """Probe every candidate shape and report the results.

    Returns:
        A process exit code: 0 if some shape answered, 1 otherwise.
    """
    load_dotenv(REPO_ROOT / ".env", override=False)
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if not key:
        print(
            "OPENROUTER_API_KEY is not set.\n"
            "  cp .env.example .env   then paste your key from "
            "https://openrouter.ai/settings/keys",
            file=sys.stderr,
        )
        return 1

    model = (os.environ.get("JEV_MODEL") or "").strip() or "typesafe/jev-1.13"
    print(f"\nProbing {model} on OpenRouter")
    print(f"key: ...{key[-4:]} (last 4 shown)\n")

    working: list[str] = []
    for label, url, payload in candidates(model):
        status, body = _post(url, payload, key)
        verdict = "OK" if 200 <= status < 300 else "no"
        print(f"[{verdict:2s}] {status:3d}  {label}")
        print(f"      {url}")
        for line in _summarise(body).splitlines()[:14]:
            print(f"      {line}")
        print()
        if 200 <= status < 300:
            working.append(f"{label}  ->  {url}")

    print("─" * 72)
    if working:
        print("Shapes that answered:")
        for entry in working:
            print(f"  • {entry}")
        print("\nIf that is not the endpoint src/jevx/client.py uses, set")
        print("JEV_DECISIONS_URL in .env to the working one.")
        return 0

    print("Nothing answered. Worth checking:")
    print("  • the key is valid and has credit (https://openrouter.ai/credits)")
    print("  • the model id is current (https://openrouter.ai/typesafe)")
    print("  • OpenRouter may expose Jev through a shape not tried here;")
    print("    add it to candidates() and re-run.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
