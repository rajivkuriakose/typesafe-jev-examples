"""The routing policy is the part you own, so it is the part under test.

Jev supplies calibrated numbers. Turning numbers into a queue is a business
decision that lives in code, gets reviewed, and changes without retraining
anything. None of these tests need a key, a network, or a model.
"""

from __future__ import annotations

import pytest

from tests.conftest import load_example

triage = load_example("01_ticket_triage")


def test_a_confident_calm_ticket_goes_to_the_standard_queue(answers):
    decision = triage.route(answers())
    assert decision.queue == "technical/standard"
    assert decision.reasons == ()


def test_an_unroutable_department_goes_to_a_human(answers):
    """Below the floor the model is genuinely unsure, so nobody gets auto-routed."""
    decision = triage.route(answers(department_confidence=0.41))
    assert decision.queue == "human-triage"
    assert "confidence" in decision.reasons[0]


def test_billing_needs_more_confidence_than_technical(answers):
    """Same confidence, different stakes: billing can move money, technical cannot."""
    shared_confidence = 0.72

    technical = triage.route(answers(department="technical", department_confidence=shared_confidence))
    billing = triage.route(answers(department="billing", department_confidence=shared_confidence))

    assert technical.queue == "technical/standard"
    assert billing.queue == "human-triage"
    assert "billing" in billing.reasons[0]


def test_confident_billing_routes_normally(answers):
    decision = triage.route(answers(department="billing", department_confidence=0.93))
    assert decision.queue == "billing/standard"


def test_a_churn_threat_outranks_the_department_queue(answers):
    decision = triage.route(answers(department="billing", department_confidence=0.95, threatens_churn=0.88))
    assert decision.queue == "retention"
    assert "churn threatened" in decision.reasons


def test_urgency_alone_does_not_earn_the_priority_queue(answers):
    """Urgent language is cheap. Only urgency plus real impact escalates."""
    decision = triage.route(answers(is_urgent=0.97))
    assert decision.queue == "technical/standard"


def test_urgency_with_business_impact_escalates(answers):
    decision = triage.route(answers(is_urgent=0.97, business_impact=1.8))
    assert decision.queue == "technical/priority"
    assert "urgent with active business impact" in decision.reasons


def test_impact_alone_does_not_earn_the_priority_queue(answers):
    decision = triage.route(answers(business_impact=1.9))
    assert decision.queue == "technical/standard"


@pytest.mark.parametrize(
    ("kwargs", "expected_note"),
    [
        ({"is_repeat_contact": 0.91}, "repeat contact"),
        ({"frustration": 1.7}, "frustration"),
    ],
)
def test_annotations_are_recorded_without_changing_the_queue(answers, kwargs, expected_note):
    """Some signals inform the agent, but do not by themselves move the ticket."""
    decision = triage.route(answers(**kwargs))
    assert decision.queue == "technical/standard"
    assert any(expected_note in reason for reason in decision.reasons)


def test_a_noul_near_one_half_is_not_treated_as_a_yes(answers):
    """0.5 means the model cannot tell, not 'moderately true'."""
    decision = triage.route(answers(threatens_churn=0.5))
    assert decision.queue != "retention"


def test_every_question_asked_is_actually_consumed_or_documented():
    """Speculative questions cost tokens; each one should earn its place."""
    asked = set(triage.QUESTIONS)
    assert asked == set(triage.QUESTION_NOTES), (
        "every question needs a line in QUESTION_NOTES explaining why it is asked"
    )
