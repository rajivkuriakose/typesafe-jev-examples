# Handoff

Everything a fresh session needs to add the next example without rediscovering
what this one cost to learn. Read this before touching code.

**Current state:** two examples, both working against live Jev. 60 offline tests
plus 1 live test, all passing. Published at
<https://github.com/rajivkuriakose/typesafe-jev-examples>, `main` tracking
`origin/main`, working tree clean.

```sh
make setup       # uv sync
make test        # 60 offline tests: no key, no network
make test-live   # 1 test, ~2 billed calls
make run         # example 01, 3 calls
make rerank      # example 02, 156 calls, ~$0.003, ~15s
make probe       # re-discover OpenRouter's endpoint
```

---

## 1. Facts that were expensive to establish

Do not re-derive these. Each one cost real time or a wrong turn.

**Jev is on OpenRouter and does not need the TypeSafe waitlist.**
`typesafe/jev-1.13` (pinned: `typesafe/jev-1.13-20260917`). $0.042/M input, **$0
output**, 32k context.

**It is not a chat model and `/chat/completions` rejects it.** The error message
is what revealed the right endpoint, and it is not in OpenRouter's docs:

```
typesafe/jev-1.13 is a decisions model and cannot be used with the
chat/completions endpoint. Use the /api/alpha/decisions endpoint instead.
```

**The working call** is TypeSafe's own System One schema posted to
`https://openrouter.ai/api/alpha/decisions`:

```json
{"model": "typesafe/jev-1.13", "state": {...}, "questions": {...}}
```

**Jev is absent from the default `/api/v1/models` listing** (445 models, no
match). Its modality is `text->decisions`, not `text->text`, and
`supported_parameters` is empty. Query
`/api/v1/models/typesafe/jev-1.13/endpoints` directly.

**`TypeSafeClient` cannot be repointed at OpenRouter with `base_url`.** It builds
its own `/v1/systemone` path — passing an OpenRouter root yields
`/api/v1/v1/systemone`. That is why `OpenRouterClient` posts directly.

**Parse with `model_validate_json`, not `model_validate`.** Score `legend` and
`probabilities` arrive with string keys and only coerce to `int` in JSON mode.

**Import `SystemOneResponse` from the package root**, not
`typesafe_sdk._schemas.models` — there are two classes with that name and only
the exported one has `.choices` / `.nouls` / `.scores`.

**Question objects serialise with `model_dump(exclude_none=True)`** and the
`type` discriminator must survive the dump. A test pins this; a silent SDK
change would otherwise fail only against the live API.

**`Score` levels are 0-based.** Three criteria means levels 0/1/2, so 1.5 sits
between middle and top.

**`TimeoutError` is not a `URLError`.** They are siblings under `OSError`.
`urllib` wraps only the connect phase, so a read timeout escapes a
`URLError`-only handler. `_read_with_retries` catches `OSError` for this reason.

**`OpenRouterClient` is safe to call from threads** — it holds only config and
each call opens its own connection. **The native `TypeSafeNativeClient` path is
not verified under concurrency** (it shares one SDK client), and is not verified
at all: nobody here has an early-access key. The README says so; keep it saying
so until someone runs it.

---

## 2. Conventions

- **uv + hatchling, src layout.** `uv sync`, `uv run`. Package is `src/jevx`.
- **Examples are numbered files** in `examples/`, imported in tests via
  `load_example("02_rerank")` from `tests/conftest.py` — the filenames are not
  valid identifiers. That loader registers the module in `sys.modules` before
  executing it, which `@dataclass` requires; do not remove that line.
- **`make test` is offline and must stay that way.** `addopts` carries
  `-m 'not live'`. Live tests are opt-in via `make test-live`.
- **Docstrings on everything public**, Args/Returns/Raises where they earn it.
  Comments explain *why*, never what changed.
- **Commit messages are prose**, not bullet lists: what changed and the reasoning
  that made it necessary. End with the `Co-Authored-By` trailer.
- **MIT.** Author is Rajiv, GitHub handle `rajivkuriakose`.

**Secrets.** The live key lives in `.env`, gitignored; `.env.example` is the
committed template. The repo is public, so before every push:

```sh
git check-ignore .env                       # must match
git diff --cached | grep -E 'sk-[A-Za-z0-9]{20,}'   # must find nothing
```

Scan history, not just the working tree — a key committed in any earlier commit
still ships.

---

## 3. The bar these examples are held to

This is the part that matters most, and the part a fresh session will otherwise
get wrong. Two review rounds were spent enforcing it.

**Policy lives in code, not in the model.** Thresholds, escalation rules and
routing are ordinary Python that can be read, tested and changed without
re-running inference. Example 01's `route()` is the reference.

**Scale thresholds to the stakes of each branch.** A single global confidence
cutoff is the common mistake. Routing to `technical` is cheap to get wrong;
routing to `billing`, which can issue a refund, is not.

**Pick the primitive by what the answer means.** One of a fixed set is a
`Choice`. Whether a condition holds is a `Noul`. A degree along a described
dimension is a `Score`. Not three flavours of classifier.

**Give the model a no-match option.** It cannot choose a value you never offered.

**Every measured claim must survive an adversarial reading.** This repo shipped a
claim that did not, and the fix is instructive:

> The re-ranking example reported "keyword top-1 67%, re-ranked 100%" and
> described the two improved queries as cases where the baseline ranked the
> answer badly. It had not. On one, the gold passage was tied *for first* and
> alphabetical tie-breaking put it fourth; on the other it scored zero alongside
> seventeen others. The baseline had expressed **no opinion**, and `sorted()`
> supplied the rank — silently converting "no signal" into "wrong answer" in the
> baseline's disfavour.

Rules that came out of that, which apply to any new example making a claim:

1. **Ties are not rankings.** If you sort by a score, report `rank_bounds`, not a
   tie-broken position. `examples/02_rerank.py` has both helpers.
2. **Apply the correction to both sides.** The first fix gave the baseline a
   range and left the star of the show a point estimate — flattering it by
   exactly the mechanism being condemned.
3. **Never quote output the program does not produce.** A README once showed a
   runtime measured by the shell, inside a block that read as a transcript. If
   you want the number, print it.
4. **Attest to the whole fixture, not the flattering half.** A provenance note
   covered how the passages were written but not the queries — and the queries
   drove the result.
5. **State what the sample size is worth.** Six queries reaching 100% is a
   demonstration, not a benchmark.

**Tests are offline and test real logic.** Pure functions get tested directly.
Stub *collaborators* are fine (`StubClient` in `tests/test_rerank.py`); mocking
the thing under test is not. Fixture integrity gets tested too — a typo in a gold
id would silently make an example unwinnable.

---

## 4. What exists, and the structural map

The examples are deliberately different *shapes*, not different topics. A new
example should add a shape, not a variation.

| | state | questions | requests |
|---|---|---|---|
| **01** ticket triage | one ticket | seven | one |
| **02** re-ranking | one article each | one | one per candidate, concurrent |

The rule underneath: **questions batch into one request only when they share
state.** Example 01 fans out questions; example 02 cannot, because each candidate
*is* different state, so it fans out requests instead.

**`src/jevx/`** is shared plumbing and should rarely change:
`open_client()` → `(client, provider)`; `client.ask(state, questions)` →
`SystemOneResponse`; `header()` / `describe()` for output. `JevClient` is the
Protocol.

---

## 5. Next: example 03, hierarchical classification

**Why this one.** It is the first example that uses the probability
*distribution* rather than just the top answer — beam search over a taxonomy,
carrying several live branches down each level. That is the idea most specific to
calibrated models and the hardest to fake with a chat model. It was deliberately
held back until example 02 gave readers distribution-reading fluency.

Cookbook: <https://docs.typesafe.ai/cookbooks/hierarchical_classification>
— **read it before designing.** Both prior examples were improved by the cookbook
disagreeing with the obvious decomposition. For 02 it revealed one `Noul` per
pair rather than one batched `Score` request, which changed the whole shape.

Also relevant: <https://docs.typesafe.ai/cookbooks/classification_using_confidence>
(uses confidence to decide *how deep* in a hierarchy to commit — a natural pairing).

**Sketch, not a design.** Run the brainstorming skill properly rather than
building from this:

- A taxonomy 3–4 levels deep. Support categories would keep continuity with 01
  and 02 and could reuse the help-center vocabulary.
- At each level, one `Choice` over that node's children. Keep the top *k*
  branches by probability rather than committing to the argmax, expand each, and
  prune.
- The teaching point is that beam search only works because the probabilities are
  calibrated and comparable — an uncalibrated model's top-1 tells you nothing
  about whether branch 2 was nearly as good.
- Confidence decides depth: commit further only while the distribution stays
  concentrated, and stop at the level where it flattens rather than guessing a
  leaf.
- Measure something honest. Accuracy at each depth, and how often the correct
  leaf was in the beam but not the argmax — that number is the argument for beam
  search over greedy, and it is the claim to be careful with.

**Shape check:** this is *many sequential rounds*, where each round's questions
depend on the previous round's answers. That is genuinely a third shape — 01 is
one round, 02 is one round fanned wide. Good.

---

## 6. Open items

- **`/api/alpha/decisions` is an alpha endpoint.** If OpenRouter moves it,
  `make probe` re-discovers the working shape and `JEV_DECISIONS_URL` in `.env`
  overrides it without a code change.
- **The native TypeSafe path has never been run.** When an early-access key
  arrives, set `TYPESAFE_API_KEY` (it takes precedence automatically), run both
  examples, and remove the "unproven" caveats from `client.py` and the README.
  `TYPESAFE_MODEL` is deliberately separate from `JEV_MODEL` because the
  namespaces are disjoint — `jev-latest` vs `typesafe/jev-1.13`.
- **`INPUT_DOLLARS_PER_TOKEN`** in `examples/02_rerank.py` is a dated external
  price and will go stale.
- **Jev returns very decisive distributions** on clear cases — confidence 1.0,
  probabilities 0/1. Thresholds tuned against a weaker model will not transfer.
  Watch for `confidence 0.00` on a `Score`: that is a flat distribution meaning
  Jev cannot tell, and it must not be acted on. Example 01 hits this live on
  ticket 3's `business_impact`.

---

## 7. Working notes

- **Run the brainstorming skill before building** and stop for approval before
  writing code. Both examples went through it and both designs changed as a
  result.
- **Request code review after**, and expect it to find something real. Both
  rounds did. Verify its claims yourself before acting — one review's arithmetic
  needed checking, and one of its findings about a test was wrong.
- **`dcg` blocks `rm -rf` even inside heredoc text.** Writing a Makefile with a
  standard `clean` target trips it. Use the file-writing tool rather than
  obfuscating the string.
- **`gh` is authenticated** as `rajivkuriakose` with `repo` scope.
- Long fan-outs hit transient upstream errors. A live 156-call run died on a
  single HTTP 520 before retries existed; assume any new fan-out needs them.
