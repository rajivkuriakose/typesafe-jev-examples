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
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from typesafe_sdk import Noul

from jevx import JevClient, MissingKeyError, ProviderError, header, open_client

CORPUS_PATH = Path(__file__).resolve().parents[1] / "data" / "help_center.json"

#: OpenRouter's published input price for typesafe/jev-1.13 as of 2026-09-18,
#: in dollars per token. Output is billed at zero. Used only to estimate the
#: cost of a run, and only when OpenRouter is the provider.
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
        All passage ids, best first. Ties break by id so runs are comparable --
        but see `rank_bounds`: a tie-broken position is not a judgement, and
        reading it as one overstates what the baseline actually said.
    """
    return rank_by_score(keyword_scores(question, passages))


def keyword_scores(question: str, passages: list[Passage]) -> dict[str, float]:
    """Score every passage by distinct query-word overlap.

    No length normalisation, so longer passages have more chances to match.
    The stopword list is also short, so common words like "have" carry weight
    they do not deserve. Both are acceptable in a baseline whose job is to be
    plainly worse than the re-ranker, but they are biases, not neutrality.

    Args:
        question: The customer's question.
        passages: Every candidate passage.

    Returns:
        Overlap count per passage id.
    """
    wanted = _terms(question)
    return {
        passage.id: float(len(wanted & _terms(f"{passage.title} {passage.text}")))
        for passage in passages
    }


def rank_by_score(scores: dict[str, float]) -> list[str]:
    """Order ids by score, highest first.

    Args:
        scores: Score per id.

    Returns:
        The ids, best first, with ties broken by id for determinism.
    """
    return sorted(scores, key=lambda key: (-scores[key], key))


def _span(low: float, high: float) -> str:
    """Render a hit-rate range, collapsing it when ties do not decide anything."""
    return f"{low:.0%}" if low == high else f"{low:.0%}–{high:.0%}"


def rank_bounds(scores: dict[str, float], target: str) -> tuple[int, int]:
    """The best and worst rank `target` could hold, given ties.

    A scorer that assigns the same value to many candidates has not ranked
    them; the sort order among them comes from the tie-break, not the scorer.
    Collapsing that to a single number turns "no opinion" into "wrong answer".

    Args:
        scores: Score per id.
        target: The id to locate.

    Returns:
        The best and worst positions, 1-based. Equal when nothing ties.
    """
    value = scores[target]
    strictly_above = sum(1 for other in scores.values() if other > value)
    tied = sum(1 for other in scores.values() if other == value)
    return strictly_above + 1, strictly_above + tied


def hit_rate_bounds(
    scores: dict[str, dict[str, float]], golds: dict[str, str], k: int
) -> tuple[float, float]:
    """Top-`k` hit rate if every tie broke against you, and if every tie broke for you.

    Args:
        scores: Score per passage id, per query id.
        golds: The correct passage id per query id.
        k: How far down the list to look.

    Returns:
        The pessimistic and optimistic hit rates.
    """
    if not scores:
        return 0.0, 0.0
    pessimistic = optimistic = 0
    for query_id, per_passage in scores.items():
        best, worst = rank_bounds(per_passage, golds[query_id])
        pessimistic += worst <= k
        optimistic += best <= k
    return pessimistic / len(scores), optimistic / len(scores)


def _score_with_usage(client: JevClient, query: Query, passage: Passage) -> tuple[float, int]:
    """Score one pair, also reporting what it cost in input tokens.

    One pair in a hundred failing anonymously is worse than useless, so both
    failure modes name the pair: a provider error, and an answer that arrives
    without the noul we asked for (`nouls` filters by type, so a mistyped
    answer shows up as a missing key rather than a wrong value).
    """
    try:
        response = client.ask(
            {
                "question": query.text,
                "article": {"title": passage.title, "text": passage.text},
            },
            {"relevant": RELEVANCE},
        )
    except ProviderError as error:
        raise ProviderError(f"{query.id}/{passage.id}: {error}") from error

    # Narrow on purpose: `nouls` filters answers by type, so an answer of the
    # wrong type is a missing key rather than a wrong value. Wrapping the call
    # above in this guard too would misreport any KeyError from inside the SDK.
    try:
        answer = response.nouls["relevant"]
    except KeyError as error:
        raise ProviderError(
            f"{query.id}/{passage.id}: answer contained no 'relevant' noul"
        ) from error
    return answer.noul, response.usage.input_tokens or 0


def score_passage(client: JevClient, query: Query, passage: Passage) -> float:
    """Ask Jev whether one passage answers one question.

    The scoring run itself goes through `_score_with_usage`, which also reports
    tokens. This is the plain single-pair entry point, for tests and for
    anyone reading the file top to bottom.

    Args:
        client: An open Jev client.
        query: The customer question.
        passage: The candidate article.

    Returns:
        The probability that the article answers the question.
    """
    score, _ = _score_with_usage(client, query, passage)
    return score


def score_corpus(client: JevClient, corpus: Corpus) -> tuple[dict[str, dict[str, float]], int]:
    """Score every (query, passage) pair, concurrently.

    Each pair is an independent request -- they cannot be batched, because
    batching shares one state across many questions and here the state differs
    per call. Concurrency is what recovers the wall-clock time instead.

    Safe to run concurrently on the OpenRouter path, where the client holds
    only configuration and every call opens its own connection. The native
    TypeSafe path shares one SDK client across threads and has not been
    exercised that way here.

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
    call_count = len(corpus.queries) * len(corpus.passages)

    try:
        client, provider = open_client()
    except MissingKeyError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1

    print(header("02 · Re-ranking", provider))
    print(
        f"{len(corpus.queries)} questions × {len(corpus.passages)} articles "
        f"= {call_count} calls, {MAX_WORKERS} at a time\n"
    )

    keyword = {query.id: keyword_scores(query.text, corpus.passages) for query in corpus.queries}
    baseline = {query_id: rank_by_score(scores) for query_id, scores in keyword.items()}

    started = time.monotonic()
    try:
        with client:
            scores, total_tokens = score_corpus(client, corpus)
    except ProviderError as error:
        print(f"{error}\n", file=sys.stderr)
        return 1
    elapsed = time.monotonic() - started

    reranked = {query_id: rank_by_score(pairs) for query_id, pairs in scores.items()}

    for query in corpus.queries:
        before_rank = baseline[query.id].index(query.gold) + 1
        after_rank = reranked[query.id].index(query.gold) + 1
        best, worst = rank_bounds(keyword[query.id], query.gold)
        arrow = "→" if before_rank == after_rank else ("↑" if after_rank < before_rank else "↓")

        print(f'{query.id}  "{query.text}"')
        print(f"  gold {query.gold} ({by_id[query.gold].title})")
        if best == worst:
            keyword_note = f"keyword rank {before_rank}"
        else:
            # A tie is the scorer declining to choose. Say so, rather than
            # letting the alphabetical tie-break masquerade as a judgement.
            tied_with = worst - best
            keyword_note = (
                f"keyword scored {keyword[query.id][query.gold]:.0f}, tied with {tied_with} "
                f"other{'s' if tied_with != 1 else ''} (rank {best}–{worst}); "
                f"id tie-break put it {before_rank}"
            )
        print(f"  {keyword_note}   {arrow}   re-ranked {after_rank}")
        for passage_id in reranked[query.id][:3]:
            mark = " ← gold" if passage_id == query.gold else ""
            print(
                f"    {scores[query.id][passage_id]:.2f}  {passage_id}  "
                f"{by_id[passage_id].title}{mark}"
            )
        print()

    print("─" * 72)
    # Both sides get the same treatment. Reporting the baseline as a range and
    # the re-ranker as a point estimate would flatter the re-ranker by exactly
    # the mechanism this example exists to warn about.
    for k in (1, 3):
        keyword_span = _span(*hit_rate_bounds(keyword, golds, k))
        reranked_span = _span(*hit_rate_bounds(scores, golds, k))
        print(f"  top-{k}   keyword {keyword_span}   re-ranked {reranked_span}")
    print("          (ranges where ties decide the outcome: a tie is not a ranking)")

    cost = (
        f", ~${total_tokens * INPUT_DOLLARS_PER_TOKEN:.4f}"
        if provider.name == "openrouter"
        else ""
    )
    print(f"\n  {call_count} calls, {total_tokens:,} input tokens{cost}, {elapsed:.1f}s")
    print(
        f"  {len(corpus.queries)} queries is a demonstration, not a benchmark. "
        "See the cookbook for measured results on a real corpus."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
