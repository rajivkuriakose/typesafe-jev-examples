"""Printing helpers, so the examples show the numbers behind every decision."""

from __future__ import annotations

from typing import Any


def header(title: str, provider: Any) -> str:
    """Return a titled banner naming the provider in use.

    Args:
        title: Example title.
        provider: The resolved `Provider`.

    Returns:
        A two-line banner.
    """
    return f"\n{title}\n{'─' * len(title)}\nprovider: {provider.label}\n"


def describe(response: Any) -> str:
    """Render every answer in a response, one per line.

    Shows the probability or score alongside confidence, because a decision
    made without looking at confidence is a decision made blind.

    Args:
        response: A `SystemOneResponse`.

    Returns:
        An indented, human-readable block.
    """
    lines: list[str] = []
    for name, answer in sorted(getattr(response, "choices", {}).items()):
        lines.append(f"  {name:22s} {answer.choice:<12s} conf {answer.confidence:.2f}")
    for name, answer in sorted(getattr(response, "nouls", {}).items()):
        lines.append(f"  {name:22s} {answer.noul:.2f}")
    for name, answer in sorted(getattr(response, "scores", {}).items()):
        confidence = getattr(answer, "confidence", None)
        suffix = f"   conf {confidence:.2f}" if isinstance(confidence, float) else ""
        lines.append(f"  {name:22s} {answer.score:.2f}{suffix}")
    return "\n".join(lines)
