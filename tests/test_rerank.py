"""Ranking, scoring and metrics are pure, so they are tested without a model.

The only thing here that needs Jev is the live test at the bottom, which is
skipped unless a key is present. Everything else -- the keyword baseline, the
ordering, the hit-rate arithmetic, and the integrity of the fixture itself --
is ordinary code and is pinned as such.
"""

from __future__ import annotations

import re
import threading
from types import SimpleNamespace

import pytest

from jevx import ProviderError

from tests.conftest import load_example

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
    assert rerank.keyword_rank("Rate limits!", corpus.passages) == rerank.keyword_rank(
        "rate limits", corpus.passages
    )


# --- turning scores into a ranking -----------------------------------------


def test_keyword_scores_covers_every_passage(corpus):
    scores = rerank.keyword_scores("payout", corpus.passages)
    assert set(scores) == {passage.id for passage in corpus.passages}


def test_rank_bounds_is_exact_when_nothing_ties():
    """A unique score pins the rank to a single position."""
    assert rerank.rank_bounds({"a": 3.0, "b": 2.0, "c": 1.0}, "b") == (2, 2)


def test_rank_bounds_spans_the_whole_tie_group():
    """Tied scores mean the rank is a range, not a number.

    This is the distinction the first version of this example got wrong: a gold
    passage tied for first was reported as rank 4 because the tie broke
    alphabetically, and that was described as the baseline ranking it badly.
    """
    assert rerank.rank_bounds({"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}, "c") == (1, 3)


def test_rank_bounds_when_everything_scores_zero():
    scores = dict.fromkeys("abcdef", 0.0)
    assert rerank.rank_bounds(scores, "f") == (1, 6)


def test_the_actual_rank_always_falls_inside_its_bounds(corpus):
    """Whatever the tie-break does, it cannot escape the bounds."""
    for query in corpus.queries:
        scores = rerank.keyword_scores(query.text, corpus.passages)
        best, worst = rerank.rank_bounds(scores, query.gold)
        actual = rerank.keyword_rank(query.text, corpus.passages).index(query.gold) + 1
        assert best <= actual <= worst


def test_hit_rate_bounds_reports_a_range_when_ties_decide_it():
    """Two queries, each tied for first. Pessimistically 0 hits, optimistically 2."""
    scores = {
        "q1": {"a": 1.0, "b": 1.0},
        "q2": {"a": 1.0, "b": 1.0},
    }
    golds = {"q1": "b", "q2": "b"}
    assert rerank.hit_rate_bounds(scores, golds, k=1) == (0.0, 1.0)


def test_hit_rate_bounds_collapses_when_there_are_no_ties():
    scores = {"q1": {"a": 1.0, "b": 0.0}}
    golds = {"q1": "a"}
    assert rerank.hit_rate_bounds(scores, golds, k=1) == (1.0, 1.0)


def test_rank_by_score_orders_high_to_low():
    assert rerank.rank_by_score({"a": 0.10, "b": 0.90, "c": 0.50}) == ["b", "c", "a"]


def test_rank_by_score_breaks_ties_deterministically():
    """Equal probabilities are common; the order must not depend on dict order."""
    assert rerank.rank_by_score({"b": 0.5, "a": 0.5, "c": 0.5}) == ["a", "b", "c"]


# --- metrics ---------------------------------------------------------------


def test_hit_rate_bounds_of_nothing_is_zero_not_a_crash():
    assert rerank.hit_rate_bounds({}, {}, k=1) == (0.0, 0.0)


# --- the question itself ---------------------------------------------------


def test_the_relevance_question_is_a_noul():
    """One binary judgement per pair, whose probability is the sort key."""
    assert rerank.RELEVANCE.type == "noul"


def test_the_question_references_both_the_query_and_the_passage():
    """State carries two named parts; the question has to name them to be answerable."""
    text = str(rerank.RELEVANCE.instructions) + str(rerank.RELEVANCE.criteria)
    assert "question" in text.lower()
    assert "article" in text.lower()


# --- concurrent scoring ----------------------------------------------------


class StubClient:
    """A stand-in provider that answers from a lookup, recording what it saw.

    A stub collaborator, not a mock of the code under test: `score_corpus`'s
    own logic -- pairing, aligning and aggregating -- runs for real.
    """

    def __init__(self, answer, tokens_each=10):
        """Answer with `answer(question_text, article_title)`."""
        self.answer = answer
        self.tokens_each = tokens_each
        self.seen: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def ask(self, state, questions):
        """Return a scripted noul for this pair."""
        assert set(questions) == {"relevant"}
        pair = (state["question"], state["article"]["title"])
        with self._lock:
            self.seen.append(pair)
        return SimpleNamespace(
            nouls={"relevant": SimpleNamespace(noul=self.answer(*pair))},
            usage=SimpleNamespace(input_tokens=self.tokens_each, output_tokens=2),
        )

    def close(self):
        """Nothing to release."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def test_score_corpus_puts_every_score_against_the_right_pair(corpus):
    """The one place a silent misalignment would corrupt every ranking.

    Each pair gets a distinct scripted value, so any crossed wire between the
    submitted pairs and the returned results shows up as a mismatch rather
    than as plausible-looking output.
    """
    titles = {passage.title: passage.id for passage in corpus.passages}
    texts = {query.text: query.id for query in corpus.queries}

    def scripted(question_text, article_title):
        query_number = int(texts[question_text][1:])
        passage_number = int(titles[article_title][1:])
        return round(query_number + passage_number / 100, 4)

    client = StubClient(scripted)
    scores, tokens = rerank.score_corpus(client, corpus)

    assert set(scores) == {query.id for query in corpus.queries}
    for query in corpus.queries:
        assert set(scores[query.id]) == {passage.id for passage in corpus.passages}
        for passage in corpus.passages:
            expected = round(int(query.id[1:]) + int(passage.id[1:]) / 100, 4)
            assert scores[query.id][passage.id] == expected


def test_score_corpus_visits_every_pair_exactly_once(corpus):
    client = StubClient(lambda question, title: 0.5)
    rerank.score_corpus(client, corpus)
    assert len(client.seen) == len(corpus.queries) * len(corpus.passages)
    assert len(set(client.seen)) == len(client.seen)


def test_score_corpus_totals_the_input_tokens(corpus):
    client = StubClient(lambda question, title: 0.5, tokens_each=7)
    _, tokens = rerank.score_corpus(client, corpus)
    assert tokens == 7 * len(corpus.queries) * len(corpus.passages)


def test_a_failing_pair_names_itself(corpus):
    """156 anonymous failures would be untriageable; the pair has to be in the message."""

    def explode(question_text, article_title):
        raise ProviderError("upstream said no")

    with pytest.raises(ProviderError) as caught:
        rerank.score_corpus(StubClient(explode), corpus)
    message = str(caught.value)
    assert re.search(r"q\d+/p\d+", message), f"no query/passage id in: {message}"
    assert "upstream said no" in message


def test_an_answer_without_the_noul_is_a_provider_error(corpus):
    """`nouls` filters by type, so a mistyped answer is a missing key, not a wrong value."""

    class Empty(StubClient):
        def ask(self, state, questions):
            return SimpleNamespace(
                nouls={}, usage=SimpleNamespace(input_tokens=1, output_tokens=1)
            )

    with pytest.raises(ProviderError, match="relevant"):
        rerank.score_corpus(Empty(lambda q, t: 0.0), corpus)


# --- live ------------------------------------------------------------------


@pytest.mark.live
def test_jev_ranks_a_paraphrased_answer_above_a_word_match(corpus):
    """q6: the answer shares no vocabulary with the question, and a decoy does.

    Keyword overlap scores the gold passage zero here and gives p13 two points,
    for "provider" and "have" -- the second being the short stopword list the
    scorer's own docstring admits to. This is the case the whole example rests
    on, not the SSO pair where the baseline was never actually fooled, so it is
    the one worth two real calls.
    """
    from jevx import open_client
    from jevx.client import MissingKeyError

    by_id = {passage.id: passage for passage in corpus.passages}
    query = next(q for q in corpus.queries if q.id == "q6")

    try:
        client, _ = open_client()
    except MissingKeyError:
        pytest.skip("needs a key; set OPENROUTER_API_KEY in .env")

    with client:
        gold = rerank.score_passage(client, query, by_id["p24"])
        decoy = rerank.score_passage(client, query, by_id["p13"])

    assert gold > decoy, f"expected p24 ({gold:.2f}) to beat p13 ({decoy:.2f})"
