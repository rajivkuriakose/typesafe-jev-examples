"""Fixtures and loaders shared by the example tests.

`load_example` imports the numbered example modules, whose filenames are not
valid identifiers. The `Fake*` dataclasses mirror the attribute surface of
`typesafe_sdk`'s response objects -- the only part of the SDK the triage policy
touches -- so that policy can be tested by handing it answers directly.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@dataclass(frozen=True)
class FakeChoice:
    """Stands in for a Choice answer."""

    choice: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeNoul:
    """Stands in for a Noul answer."""

    noul: float


@dataclass(frozen=True)
class FakeScore:
    """Stands in for a Score answer."""

    score: float
    confidence: float = 1.0
    probabilities: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class FakeResponse:
    """Stands in for a SystemOneResponse, exposing only the typed views."""

    choices: dict[str, FakeChoice]
    nouls: dict[str, FakeNoul]
    scores: dict[str, FakeScore]


def load_example(name: str):
    """Import an example module by filename, since the names are not identifiers.

    Args:
        name: Example module stem, such as `01_ticket_triage`.

    Returns:
        The imported module.
    """
    if str(EXAMPLES) not in sys.path:
        sys.path.insert(0, str(EXAMPLES))
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    # Register before executing: @dataclass resolves sys.modules[cls.__module__]
    # while the class body is being processed, and fails if it is absent.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def answers():
    """Build a response for the triage questions, overriding only what matters.

    Returns:
        A factory taking keyword overrides and returning a `FakeResponse`.
    """

    def make(
        *,
        department: str = "technical",
        department_confidence: float = 0.90,
        frustration: float = 0.2,
        business_impact: float = 0.2,
        is_urgent: float = 0.05,
        is_repeat_contact: float = 0.05,
        threatens_churn: float = 0.02,
        refund_requested: float = 0.01,
    ) -> FakeResponse:
        return FakeResponse(
            choices={"department": FakeChoice(department, department_confidence)},
            nouls={
                "is_urgent": FakeNoul(is_urgent),
                "is_repeat_contact": FakeNoul(is_repeat_contact),
                "threatens_churn": FakeNoul(threatens_churn),
                "refund_requested": FakeNoul(refund_requested),
            },
            scores={
                "frustration": FakeScore(frustration),
                "business_impact": FakeScore(business_impact),
            },
        )

    return make
