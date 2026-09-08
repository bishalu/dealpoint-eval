# How the live app would plug into Braintrust (design note, nothing built)

**Status: a design, not a description.** There is no `dealpoint/api/` and no `web/` in this repo
today. This note is written against the brief's deal page (`specs/grilled-product-brief.md` §1.4)
and API contract (§1.5), using only code that already exists: `dealpoint/agent/tools.py`'s
`@_traced` tools and `dealpoint/llm/client.py`'s `_maybe_wrap_openai`. M9d (`dealpoint/eval/
braintrust_showroom.py`) builds the free org-level scaffolding (baseline, aggregate scores, the
regressions dataset, log tags) this plan lands on; see that module's docstring for the showroom
side of the same project.

## 1. One initialisation

`braintrust.init_logger(project=PROJECT)` once, at FastAPI startup, behind an env flag (e.g.
`BRAINTRUST_LIVE_TRACING=1`). Nothing else needs to change in the agent code:

- `dealpoint/agent/tools.py:27` `_traced(**span_kwargs)` already wraps `search_agreement`,
  `get_section` and `lookup_defined_term` with `braintrust.traced(...)` when the `braintrust`
  package is importable (lines 64, 80, 91). It catches only `ImportError`.
- `dealpoint/llm/client.py:134` `_maybe_wrap_openai(client)` already applies
  `braintrust.wrap_openai` to `OpenRouterClient`'s underlying client (constructor, ~line 228),
  giving every LLM call real token counts and cost.

So `POST /api/run` produces a full span tree -- agent span, tool-call spans, LLM-call spans with
real usage -- the moment `init_logger` has run once in the process, with zero further
instrumentation in `dealpoint/agent/` or `dealpoint/llm/`.

**The two seams do not activate on the same condition, and that matters for the env flag.**
`_traced` decorates the three tools at **module import time**, gated only on whether `braintrust`
is *installed* -- with no key and no current logger, the span resolves to the SDK's no-op span and
the tool behaves exactly as before. `_maybe_wrap_openai` additionally requires
`braintrust_adapter.braintrust_available()`: importable **and** a key loadable. So today the tools
are permanently (harmlessly) instrumented and the LLM client is not; the only thing that turns a
no-op span tree into a real one is whether something in the process called `init_logger` with a
key. Because `_traced`'s decision is baked in at import time, the env flag that gates
`init_logger` at startup is **process-wide by construction** -- there is no way to flip live
tracing on or off per request without restarting the process (or re-importing the module, which
FastAPI does not do).

## 2. Every API route of brief §1.5, and whether it touches Braintrust

| route | touches Braintrust? |
|---|---|
| `GET /api/agreements` | no -- static corpus metadata |
| `GET /api/agreements/{id}/text` | no -- canonical text + section map from disk |
| `GET /api/questions` | no -- the fixed 12-question list |
| `GET /api/cases/{agreement_id}/{question_id}` | reads a **cached** trace link only: returns `braintrust_url\|null` alongside the finding, scores and gold. The brief already marks this link "may be expired; supplementary" -- it is never re-resolved live. |
| `POST /api/run` | **yes, the one route that produces a live trace.** One `init_logger` call (already running, from §1) gives the agent's tool calls and LLM calls a real span tree; the response returns the root span's permalink from the SDK's own return value, **never a guessed URL** (see §4). |
| `GET /api/market/{question_id}` | no -- expert-label distribution from `data/reports/` |
| `GET /api/reports` | offline by design (reads `data/reports/*.json`); **only when Braintrust is reachable** does it additionally fetch D19's three aggregate scores and D18's baseline deltas over REST, cached (see §6). |

## 3. Tags on a live trace

Every trace `POST /api/run` produces carries:

- `category:production` -- distinguishes it from every showroom category (`judged`, `agent`,
  `retrieval`, `prompt-variant`, the six representative categories, `live-replay-*`).
- `system:D`, `model:<the default arm-D model>` -- the deal page only ever runs the production
  system, D18's control's opposite end.
- the mirror's `case_id` and `question_id` metadata, so a production trace is joinable to the same
  gold answer and MAUD annotations every showroom trace is joinable to.

`metadata_mirror`'s `comparable` flag (`dealpoint/eval/braintrust_showroom.py`, `CANONICAL_
CATEGORIES = ("judged", "agent")`) is `0` for anything outside those two categories, so
`category:production` traces are automatically excluded from every frozen comparison, Pattern or
chart that ranks variants against the slate -- no separate filter needed. A dedicated "Production"
Logs view (`tags includes 'category:production'`, the same BTQL D21 uses for Cap-hits/Regressions)
keeps production traces visible on their own, the same way D21's two saved views work.

## 4. The trace link on the deal page

`POST /api/run`'s SDK span object returns its own permalink (`span.link()` or the logger's
export, depending on SDK version at build time) -- the response embeds that string verbatim. The
route never constructs a Braintrust URL by hand from a project/org name, matching the brief's
"may be expired; supplementary" framing: if the trace is later purged, the link 404s gracefully
and the deal page's own deterministic scores and evidence panel are unaffected.

## 5. The online rule widens to production

Today the project's one online-scoring automation is a `project_score` of type `online`
(`config.online.btql_filter = "metadata.category ILIKE 'live-replay%'"`, confirmed live
2026-09-08) -- it fires `judge-professional` on the showroom's manual `--replay` traces. Widening
its `btql_filter` to include `metadata.category = 'production'` (or, once D21's tags exist,
switching it to a `tags`-based filter such as `tags includes 'category:production'`) makes every
run the deal page triggers get a `judge-professional` score within about a minute of the trace
landing, at no code change -- **this rule was created in the UI and stays a UI edit**; nothing in
`braintrust_showroom.py` creates or edits it. The deal page's own response already carries the
deterministic `obj/*` scores computed on the spot (brief §1.4 item 5); the online rule's judge
score arrives asynchronously and is shown beside them once available, not blocking the response.

## 6. "Add to regressions" on the deal page

One dataset insert, in exactly D20's row shape: `braintrust.init_dataset(project=PROJECT, name=
"maud-dealpoint-regressions").insert(id=f"{case_id}|production-{trace_root_id}", input=case_id,
expected={"answer": gold_answer, "gold_spans": gold_spans}, metadata={"reason": "<the lawyer's
note, or blank>", "source_variant": "production", "source_log_id": trace_root_id, "rule": "human_
flagged", ...})`. Because the row shape is identical to `regression_rows()`'s output, a
human-flagged production failure and a rule-found showroom failure land in the same dataset and
read the same way in the UI -- the plan's "a bad trace becomes a test case in one click," now
closed for both the showroom and the live app.

`GET /api/reports` (§2, offline by design) reads `data/reports/*.json` for the four static tables
the brief's `/experiments` page needs. When a Braintrust key is configured and reachable, the
route additionally fetches the project's baseline (`GET /v1/project` → `settings.
baseline_experiment_id`, D18) and the three aggregate scores (`GET /v1/project_score`, D19) over
REST and caches the result -- Braintrust augments the offline report; it is never a hard
dependency of the page rendering.

## 7. Costs

- Production traces themselves are processed data only (Braintrust's free tier), same as every
  showroom log.
- The online rule's judge call is the only metered score touched by the live app: one
  `judge-professional` call, cents, once per run -- the same budget shape D19's opt-in
  `--score-trust-triple` refuses by default and the showroom never spends without asking.
- Nothing else writes a score: `POST /api/run`'s own deterministic scores are computed in Python
  on the spot and returned in the response, not posted to Braintrust as `scores`.

## 8. What stays out

- **The Braintrust AI proxy** -- optional later, purely as a judge-call cache; not needed for a
  single online rule firing once per run.
- **Braintrust tools** -- they would need the local retrieval index exposed as a Braintrust
  function; `search_agreement`/`get_section`/`lookup_defined_term` stay Python functions the agent
  calls directly.
- **Any second tracing SDK.** `braintrust.traced`/`wrap_openai` are the only instrumentation; no
  OpenTelemetry exporter or a second observability vendor is introduced for this milestone.

See also: `docs/demo-tour.md` stop 5 ("What deterministic truth captures, and scoring in
production") for the walking-tour framing of the same online-rule seam.
