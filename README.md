<div align="center">

<img src="assets/logo.svg" width="96" alt="Apprentice logo">

# Apprentice

**The apprentice watches the expensive model work. Then earns the job.**

[![website](https://img.shields.io/badge/runapprentice.com-visit-EDE6D6)](https://runapprentice.com)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![sdk](https://img.shields.io/badge/pip-runapprentice-3775A9)](https://pypi.org/project/runapprentice/)
[![built with](https://img.shields.io/badge/built%20with-Codex-412991)](https://openai.com/codex)
[![models](https://img.shields.io/badge/GPT--5.6-sol%20%C2%B7%20luna-10A37F)](https://platform.openai.com/docs/models)
[![every number](https://img.shields.io/badge/every%20number-measured-orange)](#the-numbers)

**OpenAI Build Week · Developer Tools**

<!-- VIDEO: replace VIDEO_ID once the film is public on YouTube -->
[![Watch the demo film](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://youtu.be/VIDEO_ID)

**[▶ Watch the demo film](https://youtu.be/VIDEO_ID)** — 2:38, the whole loop, ending on the
eval gate refusing a model.

</div>

---

## About this repository

**The production codebase is a private monorepo. This repository is the Build Week
artifact, and it is public.**

Saying that plainly up front, because the alternative would be to imply this is the whole
system, and it is not. What is here is real and runnable:

- the **feature we shipped during the submission window**, running standalone,
- the **benchmark that reproduces every GPT-5.6 number we claim**, with nothing of ours
  in the loop,
- an honest account of **how Codex was used**, and which work is new this week.

The product itself is live. You do not need our source to test it: sign in at
[runapprentice.com](https://runapprentice.com), `pip install runapprentice`, and the
console will show your own traffic. Instructions below.

## What is in here

| Piece | What it does | Needs |
|---|---|---|
| [`drift-demo/`](drift-demo) | **The feature built during the window**, running standalone: the drift endpoints and the console page, over a seeded SQLite database. `seed.py` writes 30 days of traffic whose feedback score decays from 0.95 to 0.55, so you can watch a model quietly get worse. | nothing |
| [`benchmark/`](benchmark) | **Reproduces the GPT-5.6 numbers.** Talks to OpenAI and DSPy directly. No Apprentice API, no account, no database. Bring a key and check our arithmetic. | `OPENAI_API_KEY` |
| [`drift-panel/`](drift-panel) | The **production source** of the two endpoints and their tests, lifted verbatim from the private monorepo, so the code being judged is the code that shipped. | reading only |

---

## What Apprentice is

Everyone can make a model cheaper. Nobody can tell you when the cheap one is *safe*.

That is the whole problem. Swapping a frontier model for a small fine-tuned one is a
two-hour job; convincing an engineering lead that quality will not quietly collapse on the
3% of inputs nobody looked at is the part that never happens. So teams keep paying frontier
prices for extraction and classification work a 4B model could do in its sleep.

Apprentice watches a task your app already runs on a frontier model, then earns the job:

1. **Capture** — two lines of SDK, and real production traffic starts landing.
2. **Verify** — rows become **gold** (a human checked it) or **silver** (deterministic
   checks passed). Everything else stays raw and never counts.
3. **Optimize** — DSPy GEPA rewrites the prompt against the verified set.
4. **Train** — a small model is fine-tuned on gold rows only.
5. **Eval gate** — every candidate is scored on held-out gold. It is promoted, or it is
   **refused**. There is no third option and no human override.
6. **Watch for drift** — after takeover, the panel shows captured traffic, the feedback
   your app reports, and offers a retrain only when enough new gold has arrived to be
   worth it.

The refuse path is the product. A model that fails the gate does not ship, and you keep
paying the model that works.

---

## Built with Codex

Codex is the implementing engineer on the backend, not an autocomplete. The process is
written into the monorepo's `AGENTS.md` and followed every time:

1. **A bounded contract, not a vibe.** The brief names the exact unit of work, the files to
   mirror, the hard guardrails, and the gate commands (mypy, ruff, pytest).
2. **Isolation.** Feature work runs in a dedicated git worktree on its own branch, so a bad
   run cannot touch `main`.
3. **Codex writes and verifies.** It implements, runs the gates itself, and reports what
   passed. It does not commit; a human does.
4. **A senior review is mandatory.** Green gates are necessary and never sufficient.

**Codex Session ID:** `019f5eb6-27b4-7e00-af1b-04285e89a907`
*(This is the Codex thread. The Devpost field calls it a "/feedback Session ID"; it has
nothing to do with this project's own `POST /v1/feedback` endpoint.)*

**Scale:** 152 Codex sessions on the monorepo between 2026-06-10 and 2026-07-14.

### The rule that earned its keep

On this very feature, Codex's tests went green and the code was still wrong three ways —
all caught by the human review, none caught by the gates:

1. The endpoint counted **gold + silver** rows toward retraining. **Training reads gold
   only**, so silver rows would never reach the model. The panel would have overstated the
   retrain payload — on a surface whose entire job is to be trusted.
2. The cutoff used the job's **queue** time instead of the moment the worker **snapshots**
   rows, so rows created during a long training run were miscounted.
3. `eligible` ignored the minimum-rows gate, so the panel would have offered a one-click
   retrain that the API rejects with a 400. A dead end, one click away from the user.

The tests passed because **they encoded the same wrong contract the brief did.** The brief
was mine, not Codex's. The lesson is that a spec must be written against the function that
enforces the rule, not against a doc that describes it.

Codex is also a good engineer inside the thread. In the session above it root-caused its
own failing test run (the reused venv's editable install pointed at the main checkout, so
pytest was exercising the old API) and caught a UTC bucketing bug in its own diff during
self-review, before handing anything back.

### GPT-5.6 inside the product

The GEPA optimizer's student and reflection models are configurable. The submission run
uses **`gpt-5.6-luna`** for both. Reproduce it yourself: [`benchmark/`](benchmark).

Codex itself runs on **`gpt-5.6-sol`** at medium effort.

---

## New this week, and what predates it

The rules require this distinction, so here it is without spin.

| | |
|---|---|
| **Built during the window (Jul 13–21)** | **The drift panel.** Two read-only endpoints and the console page that reads them. It wires the previously orphaned `POST /v1/feedback` signal into a surface a user can act on, and closes the loop after a model takes over. Written by driving Codex; see the session above. |
| **Also in-window** | The first scored GEPA run on `gpt-5.6-luna`, and a real bug it exposed: our run-cost accounting was billing for DSPy **cache hits** — LLM calls that never reached the provider — overstating cost ~15x. Fixed, and the run below is the verification. |
| **Predates the window** | Everything else: capture, the tier system, the GEPA optimizer, fine-tuning, the eval gate, the console, the SDK, the docs site. Apprentice has been in development since June 2026. |

---

## The numbers

Measured on 2026-07-14. Nothing here is projected.

| | |
|---|---|
| Baseline prompt (held out) | **50.17** |
| GEPA-optimized (held out) | **100.00** |
| Rows improved / regressed | **12 of 12 / 0** |
| Wall time | **67 seconds** |
| Cost | **$0.047** |
| Models | `gpt-5.6-luna`, student and reflection |
| Metric | deterministic JSON field F1 — no LLM judge marks its own homework |
| Split | 24 rows, seed 42, 12-row holdout |

Reproduce it: [`benchmark/`](benchmark). It talks to OpenAI and DSPy directly and needs
nothing of ours.

**Expect the shape, not the digits.** GEPA is stochastic and the baseline prompt is
deliberately weak, so the starting score moves between runs (we have seen 50.2 and 56.0).
What is stable is that the optimized prompt scores 100 on the holdout.

---

## For judges: how to test this

### 1. The drift panel, running (one command)

```bash
cd drift-demo
uv run python seed.py          # 30 days of traffic, feedback decaying 0.95 -> 0.55
uv run uvicorn app:app         # http://localhost:8000
```

The feature shipped this week, standalone, over a seeded SQLite database. You will see
captured traffic, a feedback score visibly falling, and a retrain card that only offers the
button when a retrain would actually clear the training gate.

### 2. Reproduce the GPT-5.6 numbers

```bash
cd benchmark
export OPENAI_API_KEY=sk-...
uv run python gepa_bench.py     # ~1 minute, a few cents
```

### 3. The live product

Sign in at [runapprentice.com](https://runapprentice.com), create a task, then:

```bash
pip install runapprentice
```

```python
from runapprentice import Apprentice

client = Apprentice(api_key="<from the console>")
trace_id = client.capture(task="support-triage", input=question, output=answer)
if trace_id:                                  # capture is fail-open
    client.feedback(trace_id, good=True)      # or good=False, or score=0.4
```

Your calls appear in the console under **Activity**, and the **Drift** tab charts them.

**Supported platforms:** Python 3.11+ (SDK, macOS/Linux/Windows), any OS for the console.

---

## Links

- **Live product:** [runapprentice.com](https://runapprentice.com)
- **Docs:** [docs.runapprentice.com](https://docs.runapprentice.com)
- **SDK:** [pypi.org/project/runapprentice](https://pypi.org/project/runapprentice/)
- **Public benchmark:** [apprentice-benchmark](https://github.com/singhabhishekkk/apprentice-benchmark)

MIT licensed. The production monorepo is private.
