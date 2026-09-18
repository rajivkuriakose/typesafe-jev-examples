"""Route a support ticket in one request, and let confidence decide who acts.

The lesson is decomposition. A single "triage this ticket" prompt to a chat
model returns prose you have to parse and cannot trust. Seven narrow questions
over the same state return calibrated numbers, and the routing policy stays in
ordinary Python where you can read it, test it, and change a threshold without
touching the model.

Every question is answered in one request, in parallel. Asking seven costs
roughly what asking one costs in wall-clock time, which is what makes the
decomposition worth doing.

Run:  make run   (or: uv run examples/01_ticket_triage.py)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from typesafe_sdk import Choice, Noul, Score

from jevx import MissingKeyError, ProviderError, describe, header, open_client

TICKETS = [
    "Hi, I've been trying to connect my Stripe account for 3 days and it keeps "
    "failing. I'm losing sales. Please help ASAP.",
    "Quick question — does the Team plan include SSO, or is that Enterprise "
    "only? No rush, just planning our rollout for next quarter.",
    "this is the fourth time i've written in. nobody reads these. cancel my "
    "account and refund the last two months or i'm filing a chargeback.",
]

QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this ticket.",
        criteria={
            "billing": "Charges, refunds, invoices, subscription changes, cancellations.",
            "technical": "Bugs, failed integrations, errors, anything that is not working.",
            "sales": "Pricing, plan comparisons, capabilities of plans the customer does not have.",
            "other": "None of the above applies.",
        },
    ),
    "frustration": Score(
        instructions="How frustrated the customer sounds.",
        criteria=[
            "Neutral or friendly; stating facts or asking a question.",
            "Visibly annoyed but still civil.",
            "Angry; strong language, threats, or accusations.",
        ],
    ),
    "business_impact": Score(
        instructions="How much the customer's own business is being harmed right now.",
        criteria=[
            "No harm; a question or a preference.",
            "Inconvenience; a workaround exists.",
            "Active revenue or operational loss while this continues.",
        ],
    ),
    "is_urgent": Noul(instructions="The message conveys urgency or time-sensitivity."),
    "is_repeat_contact": Noul(
        instructions="The customer says they have contacted support about this before.",
    ),
    "threatens_churn": Noul(
        instructions="The customer threatens to cancel, refund, charge back, or leave.",
    ),
    "refund_requested": Noul(
        instructions="The customer explicitly asks for a refund or a chargeback.",
    ),
}

#: Why each question is asked. Speculative questions cost tokens, so each one
#: has to earn its place. Kept in sync with QUESTIONS by a test.
QUESTION_NOTES = {
    "department": "Selects the queue.",
    "frustration": "Annotation for the agent; does not move the ticket by itself.",
    "business_impact": "Half of the priority rule; meaningless without urgency.",
    "is_urgent": "Half of the priority rule; meaningless without impact.",
    "is_repeat_contact": "Annotation; a second miss is worse than a first.",
    "threatens_churn": "Overrides the department queue and sends the ticket to retention.",
    "refund_requested": "Speculative: read only on the billing branch, asked of every ticket.",
}

# ---------------------------------------------------------------------------
# Policy. These numbers are decisions, not model output. Tune them on your own
# tickets; the docs' 0.5 / 0.9 bands are a starting point, not a law.
# ---------------------------------------------------------------------------

#: Below this, the model cannot tell which team owns the ticket at all.
ROUTABLE_CONFIDENCE = 0.60

#: Billing can issue refunds, so it needs more certainty than a queue move.
BILLING_CONFIDENCE = 0.85

#: A Noul at 0.5 means "cannot tell", not "moderately true". Stay well above it.
LIKELY = 0.70

#: Scores are probability-weighted, so 1.5 sits between the middle and top level.
IMPACT_THRESHOLD = 1.5
FRUSTRATION_THRESHOLD = 1.6


@dataclass(frozen=True)
class Decision:
    """A routing outcome and the evidence behind it."""

    queue: str
    reasons: tuple[str, ...]


def route(response) -> Decision:
    """Turn typed answers into one routing decision.

    Args:
        response: A System One response for a single ticket.

    Returns:
        The queue to route to, and the reasons that produced it.
    """
    department = response.choices["department"]
    reasons: list[str] = []

    if department.confidence < ROUTABLE_CONFIDENCE:
        return Decision(
            "human-triage",
            (f"department confidence {department.confidence:.2f} below {ROUTABLE_CONFIDENCE}",),
        )

    # A churn threat is department-independent and outranks the normal queue.
    if response.nouls["threatens_churn"].noul > LIKELY:
        return Decision("retention", ("churn threatened",))

    if department.choice == "billing" and department.confidence < BILLING_CONFIDENCE:
        return Decision(
            "human-triage",
            (
                f"billing needs {BILLING_CONFIDENCE} confidence to auto-route, "
                f"got {department.confidence:.2f}",
            ),
        )

    queue = f"{department.choice}/standard"
    urgent = response.nouls["is_urgent"].noul > LIKELY
    impacted = response.scores["business_impact"].score >= IMPACT_THRESHOLD
    if urgent and impacted:
        queue = f"{department.choice}/priority"
        reasons.append("urgent with active business impact")

    # Annotations: useful to the agent, but they do not move the ticket.
    if department.choice == "billing" and response.nouls["refund_requested"].noul > LIKELY:
        reasons.append("refund requested")
    if response.nouls["is_repeat_contact"].noul > LIKELY:
        reasons.append("repeat contact")
    frustration = response.scores["frustration"].score
    if frustration >= FRUSTRATION_THRESHOLD:
        reasons.append(f"frustration {frustration:.2f}")

    return Decision(queue, tuple(reasons))


def main() -> int:
    """Triage every sample ticket, printing the answers and the decision.

    Returns:
        A process exit code.
    """
    try:
        client, provider = open_client()
    except MissingKeyError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1

    print(header("01 · Ticket triage", provider))
    try:
        with client:
            for ticket in TICKETS:
                response = client.ask({"ticket": ticket}, QUESTIONS)
                decision = route(response)
                print(f'"{ticket[:70]}..."')
                print(describe(response))
                trailer = f"   ({'; '.join(decision.reasons)})" if decision.reasons else ""
                print(f"  → route: {decision.queue}{trailer}")
                print(f"  tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out\n")
    except ProviderError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
