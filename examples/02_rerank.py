"""Re-rank search results by asking one question about each candidate.

Example 01 asks many questions about one thing, in one request. This is the
mirror image: one question about many things, which cannot share a request
because each candidate *is* different state. So the calls fan out concurrently
instead, and the probability that comes back is the sort key.

That shape is what makes a decision model useful inside a retrieval pipeline.
Keyword search is fast and cheap but matches words; it cannot tell that someone
asking "there is no SSO option in settings" needs the pricing page rather than
the SSO setup guide. Ranking the shortlist by a judgement fixes the top of the
list, which is the only part anyone reads.

Run:  make rerank   (or: uv run examples/02_rerank.py)
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import Noul

from jevx import MissingKeyError, ProviderError, header, open_client

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "help_center.json"

#: OpenRouter's published input price for typesafe/jev-1.13, in dollars per
#: token. Output is billed at zero. Used only to estimate the cost of a run.
INPUT_DOLLARS_PER_TOKEN = 0.042 / 1_000_000

#: Concurrent requests. Each pair is an independent call, so this is just how
#: hard we are willing to lean on the API at once.
MAX_WORKERS = 8

#: Words carrying no retrieval signal, removed before overlap scoring.
STOPWORDS = frozenset(
    "a an and are as at be but by for from how i if in is it its my of on or our "
    "that the their there they this to up was we what when where which who why "
    "will with you your".split()
)


@dataclass(frozen=True)
class Passage:
    """One help-center article."""

    id: str
    title: str
    text: str


@dataclass(frozen=True)
class Query:
    """A customer question, and the article that actually answers it."""

    id: str
    text: str
    gold: str


@dataclass(frozen=True)
class Corpus:
    """The articles to search and the questions to search them with."""

    passages: list[Passage]
    queries: list[Query]


#: One binary judgement per (question, article) pair. A Noul rather than a Score
#: because there is no rubric to place a candidate on -- either the article
#: answers the question or it does not -- and its probability sorts directly.
RELEVANCE = Noul(
    instructions=(
        "The customer asked `question` and this support article `article` was retrieved. "
        "Does the article actually answer what the customer asked?"
    ),
    criteria={
        "true": (
            "The article resolves the customer's situation, including when it does so by "
            "explaining why the thing they want is unavailable to them."
        ),
        "false": (
            "The article is on a related topic or shares vocabulary with the question, but "
            "does not tell the customer what to do about their situation."
        ),
    },
)


def load_corpus(path: Path = CORPUS_PATH) -> Corpus:
    """Read the shipped fixture.

    Args:
        path: Location of the corpus JSON.

    Returns:
        The parsed corpus.
    """
    raw = json.loads(path.read_text())
    return Corpus(
        passages=[Passage(**entry) for entry in raw["passages"]],
        queries=[Query(**entry) for entry in raw["queries"]],
    )


def _terms(text: str) -> set[str]:
    """Split text into lowercase content words."""
    return {word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS}


def keyword_rank(question: str, passages: list[Passage]) -> list[str]:
    """Rank passages by how many distinct query words they contain.

    A deliberately plain baseline -- not BM25, and named so nobody mistakes it
    for one. It stands in for the fast retrieval stage that a re-ranker sits
    behind, and it fails in exactly the way word matching always fails: it
    cannot tell a shared vocabulary from a shared meaning.

    Args:
        question: The customer's question.
        passages: Every candidate passage.

    Returns:
        All passage ids, best first. Ties break by id so runs are comparable.
    """
    wanted = _terms(question)
    scored = {
        passage.id: len(wanted & _terms(f"{passage.title} {passage.text}")) for passage in passages
    }
    return rank_by_score(scored)


def rank_by_score(scores: dict[str, float]) -> list[str]:
    """Order ids by score, highest first.

    Args:
        scores: Score per id.

    Returns:
        The ids, best first, with ties broken by id for determinism.
    """
    return sorted(scores, key=lambda key: (-scores[key], key))


def hit_at_k(ranking: list[str], gold: str, k: int) -> bool:
    """Whether the right answer appears in the first `k` results."""
    return gold in ranking[:k]


def hit_rate(rankings: dict[str, list[str]], golds: dict[str, str], k: int) -> float:
    """Fraction of queries whose right answer lands in the top `k`.

    Args:
        rankings: Ranked passage ids per query id.
        golds: The correct passage id per query id.
        k: How far down the list to look.

    Returns:
        A fraction between 0 and 1, or 0.0 when there are no queries.
    """
    if not rankings:
        return 0.0
    hits = sum(1 for query_id, ranking in rankings.items() if hit_at_k(ranking, golds[query_id], k))
    return hits / len(rankings)


def _score_with_usage(client, query: Query, passage: Passage) -> tuple[float, int]:
    """Score one pair, also reporting what it cost in input tokens."""
    response = client.ask(
        {
            "question": query.text,
            "article": {"title": passage.title, "text": passage.text},
        },
        {"relevant": RELEVANCE},
    )
    return response.nouls["relevant"].noul, response.usage.input_tokens or 0


def score_passage(client, query: Query, passage: Passage) -> float:
    """Ask Jev whether one passage answers one question.

    Args:
        client: An open Jev client.
        query: The customer question.
        passage: The candidate article.

    Returns:
        The probability that the article answers the question.
    """
    score, _ = _score_with_usage(client, query, passage)
    return score


def score_corpus(client, corpus: Corpus) -> tuple[dict[str, dict[str, float]], int]:
    """Score every (query, passage) pair, concurrently.

    Each pair is an independent request -- they cannot be batched, because
    batching shares one state across many questions and here the state differs
    per call. Concurrency is what recovers the wall-clock time instead.

    Args:
        client: An open Jev client.
        corpus: The passages and queries.

    Returns:
        Scores keyed by query id then passage id, and the total input tokens.
    """
    pairs = [(query, passage) for query in corpus.queries for passage in corpus.passages]
    scores: dict[str, dict[str, float]] = {query.id: {} for query in corpus.queries}
    total_tokens = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = pool.map(lambda pair: _score_with_usage(client, *pair), pairs)
        for (query, passage), (score, tokens) in zip(pairs, results, strict=True):
            scores[query.id][passage.id] = score
            total_tokens += tokens

    return scores, total_tokens


def main() -> int:
    """Rank the fixture with keywords, re-rank it with Jev, and compare.

    Returns:
        A process exit code.
    """
    corpus = load_corpus()
    by_id = {passage.id: passage for passage in corpus.passages}
    golds = {query.id: query.gold for query in corpus.queries}

    try:
        client, provider = open_client()
    except MissingKeyError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1

    print(header("02 · Re-ranking", provider))
    print(
        f"{len(corpus.queries)} questions × {len(corpus.passages)} articles "
        f"= {len(corpus.queries) * len(corpus.passages)} calls, {MAX_WORKERS} at a time\n"
    )

    baseline = {query.id: keyword_rank(query.text, corpus.passages) for query in corpus.queries}
    call_count = len(corpus.queries) * len(corpus.passages)
    try:
        with client:
            scores, total_tokens = score_corpus(client, corpus)
    except ProviderError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1

    reranked = {query_id: rank_by_score(pairs) for query_id, pairs in scores.items()}

    for query in corpus.queries:
        before = baseline[query.id]
        after = reranked[query.id]
        moved = before.index(query.gold) + 1, after.index(query.gold) + 1
        verdict = "→" if moved[0] == moved[1] else ("↑" if moved[1] < moved[0] else "↓")
        print(f'{query.id}  "{query.text[:62]}..."')
        print(f"  gold {query.gold} ({by_id[query.gold].title})")
        print(f"  keyword rank {moved[0]}   {verdict}   re-ranked {moved[1]}")
        for passage_id in after[:3]:
            mark = " ← gold" if passage_id == query.gold else ""
            print(
                f"    {scores[query.id][passage_id]:.2f}  {passage_id}  "
                f"{by_id[passage_id].title}{mark}"
            )
        print()

    print("─" * 72)
    for k in (1, 3):
        before = hit_rate(baseline, golds, k)
        after = hit_rate(reranked, golds, k)
        print(f"  top-{k}   keyword {before:.0%}   re-ranked {after:.0%}")

    estimated = total_tokens * INPUT_DOLLARS_PER_TOKEN
    print(f"\n  {call_count} calls, {total_tokens:,} input tokens, ~${estimated:.4f}")
    print(
        f"  {len(corpus.queries)} queries is a demonstration, not a benchmark. "
        "See the cookbook for measured results on a real corpus."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
