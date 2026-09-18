"""Ranking, scoring and metrics are pure, so they are tested without a model.

The only thing here that needs Jev is the live test at the bottom, which is
skipped unless a key is present. Everything else -- the keyword baseline, the
ordering, the hit-rate arithmetic, and the integrity of the fixture itself --
is ordinary code and is pinned as such.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.conftest import load_example

# The live test below is gated on a key being visible. pytest does not read
# .env on its own, so load it here -- the offline tests pass explicit
# environments and are unaffected either way.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

rerank = load_example("02_rerank")


@pytest.fixture(scope="module")
def corpus():
    """The shipped help-center fixture."""
    return rerank.load_corpus()


# --- fixture integrity -----------------------------------------------------


def test_every_query_points_at_a_passage_that_exists(corpus):
    """A typo in a gold id would silently make the example unwinnable."""
    ids = {passage.id for passage in corpus.passages}
    for query in corpus.queries:
        assert query.gold in ids, f"{query.id} cites missing passage {query.gold}"


def test_passage_ids_are_unique(corpus):
    ids = [passage.id for passage in corpus.passages]
    assert len(ids) == len(set(ids))


def test_the_corpus_is_big_enough_to_be_worth_ranking(corpus):
    assert len(corpus.passages) >= 20
    assert len(corpus.queries) >= 5


# --- the keyword baseline --------------------------------------------------


def test_keyword_rank_puts_literal_overlap_first(corpus):
    """The baseline is a word-overlap scorer; this is what it is good at."""
    ranking = rerank.keyword_rank("rotating and revoking API keys", corpus.passages)
    assert ranking[0] == "p18"


def test_keyword_rank_returns_every_passage_exactly_once(corpus):
    ranking = rerank.keyword_rank("payout", corpus.passages)
    assert sorted(ranking) == sorted(passage.id for passage in corpus.passages)


def test_keyword_rank_is_deterministic(corpus):
    """Ties must break the same way every run, or the baseline is noise."""
    first = rerank.keyword_rank("account", corpus.passages)
    second = rerank.keyword_rank("account", corpus.passages)
    assert first == second


def test_keyword_rank_ignores_case_and_punctuation(corpus):
    assert rerank.keyword_rank("Rate limits!", corpus.passages)[0] == rerank.keyword_rank(
        "rate limits", corpus.passages
    )[0]


# --- turning scores into a ranking -----------------------------------------


def test_rank_by_score_orders_high_to_low():
    assert rerank.rank_by_score({"a": 0.10, "b": 0.90, "c": 0.50}) == ["b", "c", "a"]


def test_rank_by_score_breaks_ties_deterministically():
    """Equal probabilities are common; the order must not depend on dict order."""
    assert rerank.rank_by_score({"b": 0.5, "a": 0.5, "c": 0.5}) == ["a", "b", "c"]


# --- metrics ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("ranking", "gold", "k", "expected"),
    [
        (["p1", "p2", "p3"], "p1", 1, True),
        (["p1", "p2", "p3"], "p2", 1, False),
        (["p1", "p2", "p3"], "p3", 3, True),
        (["p1", "p2", "p3"], "p3", 2, False),
    ],
)
def test_hit_at_k(ranking, gold, k, expected):
    assert rerank.hit_at_k(ranking, gold, k) is expected


def test_hit_rate_is_the_fraction_of_queries_that_hit():
    rankings = {"q1": ["a", "b"], "q2": ["b", "a"], "q3": ["b", "a"]}
    golds = {"q1": "a", "q2": "a", "q3": "b"}
    # q1 and q3 have their gold first; q2 does not.
    assert rerank.hit_rate(rankings, golds, k=1) == pytest.approx(2 / 3)
    assert rerank.hit_rate(rankings, golds, k=2) == pytest.approx(1.0)


def test_hit_rate_of_nothing_is_zero_not_a_crash():
    assert rerank.hit_rate({}, {}, k=1) == 0.0


# --- the question itself ---------------------------------------------------


def test_the_relevance_question_is_a_noul():
    """One binary judgement per pair, whose probability is the sort key."""
    assert rerank.RELEVANCE.type == "noul"


def test_the_question_references_both_the_query_and_the_passage():
    """State carries two named parts; the question has to name them to be answerable."""
    text = str(rerank.RELEVANCE.instructions) + str(rerank.RELEVANCE.criteria)
    assert "question" in text.lower()
    assert "article" in text.lower()


# --- live ------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TYPESAFE_API_KEY")),
    reason="needs a key; set OPENROUTER_API_KEY in .env",
)
def test_jev_prefers_the_gold_passage_over_a_keyword_decoy(corpus):
    """The q4 case: 'SSO' points at the setup article, but the answer is the plan page.

    This is the single pair that motivates the whole example, so it is worth one
    real call each way.
    """
    from jevx import open_client

    by_id = {passage.id: passage for passage in corpus.passages}
    query = next(q for q in corpus.queries if q.id == "q4")

    client, _ = open_client()
    with client:
        gold = rerank.score_passage(client, query, by_id["p10"])
        decoy = rerank.score_passage(client, query, by_id["p11"])

    assert gold > decoy, f"expected p10 ({gold:.2f}) to beat p11 ({decoy:.2f})"
