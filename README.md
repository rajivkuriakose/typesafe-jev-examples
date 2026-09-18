# typesafe-jev-examples

Worked examples for **Jev**, [TypeSafe](https://typesafe.ai)'s first *System One*
model — runnable today through OpenRouter, without an early-access key.

System One models do not generate prose. You send **state** and a map of named
**questions**; you get back typed answers with calibrated probabilities. There is
no parsing step and no free-text-to-struct failure mode, so the interesting work
moves out of prompt engineering and into ordinary code.

```python
answer.choices["department"].choice       # "technical"
answer.choices["department"].confidence   # 0.74
answer.nouls["is_urgent"].noul            # 0.97
answer.scores["frustration"].score        # 1.35  (between "annoyed" and "angry")
```

## Two ways to reach Jev

TypeSafe's direct API is behind an early-access waitlist. OpenRouter serves the
same model today:

| | Model id | Price |
|---|---|---|
| **OpenRouter** | `typesafe/jev-1.13` | $0.042/M input, **$0** output |
| | `typesafe/jev-1.13-20260917` (pinned) | |
| **TypeSafe direct** | `jev-latest` | see typesafe.ai |

Both speak the same protocol and return the same typed response, so nothing above
the client knows which answered. [`src/jevx/client.py`](src/jevx/client.py) picks a
provider from your environment:

```
TYPESAFE_API_KEY set     ->  api.typesafe.ai          (wins if present)
OPENROUTER_API_KEY set   ->  OpenRouter, typesafe/jev-1.13
```

Everything here is verified against the OpenRouter path. The direct path is
written against the SDK's documented surface but has not been run — it needs an
early-access key. Each provider reads its own model override, `JEV_MODEL` and
`TYPESAFE_MODEL`, because the two namespaces are disjoint.

## Quickstart

```sh
git clone https://github.com/rajivkuriakose/typesafe-jev-examples
cd typesafe-jev-examples
make setup

cp .env.example .env        # then paste your key into .env
make test                   # offline: no key, no network
make run                    # triages three sample tickets against Jev
```

Get an OpenRouter key at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys).
`.env` is gitignored and will not be committed; `.env.example` is the template.

## Example 01 · Ticket triage

[`examples/01_ticket_triage.py`](examples/01_ticket_triage.py) routes a support
ticket using seven narrow questions answered in a **single parallel request**.
Asking seven costs roughly what asking one costs in wall-clock time, which is
what makes the decomposition worth doing.

```python
QUESTIONS = {
    "department": Choice(
        instructions="Which team should handle this ticket.",
        criteria={
            "billing":   "Charges, refunds, invoices, subscription changes.",
            "technical": "Bugs, failed integrations, anything not working.",
            "sales":     "Pricing, plan comparisons, plans they do not have.",
            "other":     "None of the above applies.",
        },
    ),
    "business_impact": Score(
        instructions="How much the customer's own business is being harmed right now.",
        criteria=[
            "No harm; a question or a preference.",
            "Inconvenience; a workaround exists.",
            "Active revenue or operational loss while this continues.",
        ],
    ),
    "threatens_churn": Noul(
        instructions="The customer threatens to cancel, refund, charge back, or leave.",
    ),
    # ...
}
```

Four ideas the example is built around:

**Pick the primitive by what the answer means.** One of a fixed set is a `Choice`.
Whether a condition holds is a `Noul`. A degree along a described dimension is a
`Score`. These are not three flavours of classifier.

**Scale the confidence threshold to the stakes.** A single global cutoff is the
common mistake. Routing to `technical` is cheap to get wrong — the ticket moves
queues once. Routing to `billing`, which can issue a refund, is not:

```python
ROUTABLE_CONFIDENCE = 0.60   # below this, nobody gets auto-routed
BILLING_CONFIDENCE  = 0.85   # this branch can move money
```

**Keep policy in code, not in the model.** Escalation is `churn > 0.70` *or*
`impact >= 1.5 and urgent > 0.70` — separate conditions, because an "any serious
signal" rule does not survive being averaged into a weighted score. Change a
threshold and nothing needs re-running.

**Give the model a way to say "none of these".** The `other` option exists
because a model cannot choose a value you never offered it.

## Example 02 · Re-ranking

[`examples/02_rerank.py`](examples/02_rerank.py) is the mirror image of 01, and
the contrast is the point:

| | state | questions | requests |
|---|---|---|---|
| **01** triage | one ticket | seven | one |
| **02** re-rank | one article each | one | one per candidate, concurrent |

Questions batch into a single request only when they share the same state. Here
every candidate *is* different state, so they cannot batch — the calls fan out
across a thread pool instead, and the returned probability is the sort key.

One `Noul` per (question, article) pair, sorted descending:

```python
RELEVANCE = Noul(
    instructions=(
        "The customer asked `question` and this support article `article` was "
        "retrieved. Does the article actually answer what the customer asked?"
    ),
    criteria={
        "true":  "The article resolves the customer's situation, including when it "
                 "does so by explaining why the thing they want is unavailable to them.",
        "false": "The article is on a related topic or shares vocabulary with the "
                 "question, but does not tell the customer what to do about it.",
    },
)
```

A `Noul` rather than a `Score` because there is no rubric to place a candidate
on — either the article answers the question or it does not — and a probability
sorts directly.

### What it measures

`make rerank` ranks 26 help-center articles against 6 customer questions with a
keyword baseline, then re-ranks them with Jev:

```
  top-1   keyword 67%   re-ranked 100%
  top-3   keyword 67%   re-ranked 100%
  156 calls, 69,474 input tokens, ~$0.0029, ~16s
```

**Read that honestly.** Six queries reaching 100% is not evidence of anything —
it is four queries the baseline already ranked first, plus two it did not:

```
q5  "our server was down for an hour, did we lose the events"
    keyword rank 4    ->  re-ranked 1
q6  "we are moving to another provider but I need everything we have first"
    keyword rank 25   ->  re-ranked 1      (of 26)
```

q6 is the honest illustration. The answer is *Exporting your data*, which shares
almost no vocabulary with the question, so word matching buried it second-to-last.
That is the failure mode a re-ranker exists to fix, and no amount of tuning the
keyword scorer addresses it.

Worth recording that a prediction failed here too: the corpus was built expecting
keyword search to be fooled by *"we need single sign-on but there is no option in
settings"* — grabbing the SSO setup guide instead of the pricing page that
explains SSO is Enterprise-only. It was not fooled; the pricing page happens to
contain both "single sign-on" and "settings". Jev still separated them (0.94 vs
0.77), but the baseline got there first.

The fixture was written before any baseline was run, and no query was changed
afterwards. It demonstrates the *pattern*; for measured results on a real corpus
see TypeSafe's [re-ranking cookbook](https://docs.typesafe.ai/cookbooks/rerank_typesafe),
which reports top-1 going 5% → 18% over 1,200 calls on legal retrieval.

## Testing

The routing policy is pure code: answers in, decision out. It is tested directly,
with no key, no network and no model.

```sh
make test        # offline: no key, no network
make test-live   # also the tests that call Jev
```

That split is the point. Model quality is measured against your own data; policy
correctness is measured by tests that run in milliseconds.

## How Jev is served on OpenRouter

Jev is a **decisions model**, not a text model, and OpenRouter serves it on a
dedicated endpoint. Sending it to `/chat/completions` returns:

```
typesafe/jev-1.13 is a decisions model and cannot be used with the
chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.
```

The working request is TypeSafe's own System One schema, posted there:

```http
POST https://openrouter.ai/api/alpha/decisions
Authorization: Bearer $OPENROUTER_API_KEY

{"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}
```

```json
{"model": "typesafe/jev-1.13-20260917",
 "answers": {"threatens_churn": {"type": "noul", "noul": 0.99}},
 "usage": {"input_tokens": 456, "output_tokens": 81, "cost": 1.9152e-05},
 "provider": "TypeSafe"}
```

Two consequences, both handled in [`src/jevx/client.py`](src/jevx/client.py):

- **The SDK cannot simply be repointed.** `TypeSafeClient` builds its own
  `/v1/systemone` path, so `base_url` alone will not reach the decisions
  endpoint. The OpenRouter path posts directly.
- **The response still parses as `SystemOneResponse`.** Both providers return
  the same typed object, so nothing above the client changes. One asymmetry:
  the SDK's internal decoder ignores answer types it does not recognise,
  whereas parsing the body directly rejects them. This path is the stricter of
  the two.

Jev also does not appear in the default `/api/v1/models` catalog listing — its
modality is `text->decisions` and `supported_parameters` is empty. Query
`/api/v1/models/typesafe/jev-1.13/endpoints` directly.

`make probe` re-runs the *endpoint* discovery against the live service, which is
useful if OpenRouter moves it out of alpha. It does not re-check the catalog
listing above.

## Layout

```
examples/01_ticket_triage.py   many questions, one state, one request
examples/02_rerank.py          one question, many states, concurrent requests
data/help_center.json          fixture corpus for 02
src/jevx/client.py             provider selection and transport
src/jevx/report.py             printing helpers
scripts/probe_openrouter.py    discover how OpenRouter serves Jev
tests/                         offline tests, plus one `live` test
```

## Reference

- [TypeSafe docs](https://docs.typesafe.ai) · [primitives](https://docs.typesafe.ai/primitives)
  · [confidence](https://docs.typesafe.ai/confidence) · [HTTP API](https://docs.typesafe.ai/api)
- [Jev on OpenRouter](https://openrouter.ai/typesafe)
- [Known limitations of Jev 1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

## License

[MIT](LICENSE)
