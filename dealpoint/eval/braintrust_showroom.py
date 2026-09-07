"""Braintrust showroom: make the existing `dealpoint-eval` project walkable tab by tab, for free.

Projects are free and scores cost money, and the scores this project already holds were paid for.
So this module never writes a score. It only adds the things Braintrust does not charge scores for:

    tag      experiment-level metadata, tags and descriptions, so the Experiments tab reads as a story
    rows     row-level metadata merged onto existing rows (arm, model, variant, case_type), so
             group-by and filters work; merge-updates on `metadata` do not touch scores
    logs     every stored agent run replayed into Logs as a full span tree with NO scores: the 108
             judged traces, the six representative ones, and the rest of the four-arm and Pareto
             sweeps (Topics, Patterns, Debugger, online scoring and Loop need Logs, not experiments)
    review   the 12-trace review set flagged for human review on the judge experiments
    params   four Parameters objects, one per arm, so "change one parameter, rerun, compare" is a real object
    prompts  three arm-A prompt variants for the Playground A/B/C, next to the base prompt
    judges   the calibrated rubric as four Braintrust LLM scorers on the OpenRouter judge models (your key,
             your ledger; creation is free, invocation is an OpenRouter call) via `bt scorers create`
    views    saved comparison views: arms, models, judges, RAG
    mirror   every number the dashboard needs (labels, obj/judge/human/DeepEval, cost, latency) merged onto
             each root log as METADATA (free), never as scores (metered): the Monitor page only sees logs
    raglogs  the M3 tournament as score-free logs (58 dev queries x 6 retrievers, gold-span rank in metadata)
    promptlogs  the Playground pre-run experiments mirrored to logs, judges' scores as metadata
    playground  the 18 judged cases as the exact arm-A packets (a dataset) + base/terse/cite-first/abstain-first
             as chat prompts, so the Playground prompt A/B/C holds everything but the system prompt fixed;
             `--live --run-playground` pre-runs it server-side (POST /v1/eval) as four experiments (scores!)

Dry run is the default and prints the plan; `--live` executes. Every live write is recorded in the
score ledger with n_scores=0, and a guard asserts no payload ever carries a `scores` key.

    uv run python -m dealpoint.eval.braintrust_showroom              # dry run
    uv run python -m dealpoint.eval.braintrust_showroom --live       # execute
    uv run python -m dealpoint.eval.braintrust_showroom --live --only tag,rows
    uv run python -m dealpoint.eval.braintrust_showroom --live --replay contract_144__q05:D@glm   # one new log, now
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import requests

from dealpoint.eval.braintrust_adapter import (
    PROJECT,
    load_braintrust_key,
    org_report_path,
    org_slug,
)

API = "https://api.braintrust.dev/v1"
PROJECT_ID = ""          # resolved by name at runtime (see Api.resolve_project); never hardcode an org's ids
OWNER_USER_ID = ""       # the project's creator, resolved alongside
ENV_FILE_DEFAULT = ".env.braintrust"
# Models run on the operator's OpenRouter key (configured as a Braintrust AI provider), never on
# Braintrust's built-in models: same ledger and provider choices as the rest of the project.
WORKHORSE_MODEL = "z-ai/glm-5.3-flash"
JUDGE_MODELS = {"reasoning": "mistralai/mistral-small-3.2-24b-instruct", "evidence": "mistralai/mistral-small-3.2-24b-instruct",
                "professional": "mistralai/mistral-small-3.2-24b-instruct", "trajectory": "bytedance-seed/seed-2.0-mini"}
ENV_FILE = os.environ.get("BRAINTRUST_ENV_FILE", ENV_FILE_DEFAULT)
# Ledger and manifest are per org: resolved lazily from the active key into
# data/reports/orgs/<org-slug>/ (see braintrust_adapter.org_report_path). A set value here
# (tests) or BRAINTRUST_LEDGER_FILE (explicit override) wins over the org-derived path.
LEDGER_PATH: Path | None = None
MANIFEST_PATH: Path | None = None
STEPS = ("tag", "rows", "logs", "mirror", "raglogs", "review", "params", "prompts", "judges", "views", "playground", "judgeplayground", "promptlogs")

_ARM = re.compile(r"^(?P<arm>[ABCD])-(?P<model>.+)-e2b4a2b97561-(?P<sha>[0-9a-f]{7})$")
_JUDGED = re.compile(r"^judged-(?P<arm>[ABCD])-(?P<model>.+)-e2b4a2b97561-(?P<sha>[0-9a-f]{7})$")
_JUDGE = re.compile(r"^judge-(?P<variant>[ABCD]@.+)$")
_PARETO = re.compile(r"^pareto-(?P<model>.+)$")
_RAG = re.compile(r"^rag-(?P<config>.+)$")
GLM = "z-ai/glm-5.3-flash"


def _model_from_slug(slug: str) -> str:
    return slug.replace("_", "/", 1)


SHORT_MODEL = {"z-ai/glm-5.3-flash": "glm", "anthropic/claude-haiku-4.5": "haiku", "deepseek/deepseek-v4-flash": "deepseek-v4-flash",
               "qwen/qwen3.7-flash": "qwen3.7-flash", "google/gemini-3.1-flash-lite": "gemini-3.1-flash-lite"}
ARM_SHAPE = {"A": ("pipeline", "dense", "off"), "B": ("agent", "dense", "off"), "C": ("agent", "hybrid_rrf", "off"), "D": ("agent", "hybrid_rrf", "on")}
ARM_STEP = {"A": "the control", "B": "differs from A by: loop (pipeline -> agent)", "C": "differs from B by: retriever (dense -> hybrid_rrf)",
            "D": "differs from C by: skill (off -> on)"}
RAG_CONFIG = {"m3-dense": "dense", "m3-bm25": "bm25", "m3-hybrid": "hybrid_rrf", "m3-hybrid-rerank": "hybrid_rrf_rerank",
              "m3-fusion": "multi_query_fusion", "m3-fusion-rerank": "multi_query_fusion_rerank",
              "m7-li-crosscheck": "li_native_bm25", "m7-synthetic": "hybrid_rrf"}


def _short(model: str) -> str:
    return SHORT_MODEL.get(model, model)


def _factorial(axis: str, *, varies: str, holds: str, cases: str, scorers: str, description: str, **fields) -> dict:
    """One schema for every experiment, so the Experiments table reads as a grid once the metadata
    columns axis / arm / loop / retriever / skill / model / cases are shown: `axis` is the question the
    experiment belongs to, `varies` the one variable that changes along that axis, `holds` what is fixed."""
    arm = fields.get("arm")
    if arm in ARM_SHAPE and "loop" not in fields:
        fields["loop"], fields["retriever"], fields["skill"] = ARM_SHAPE[arm]
    tags = [f"axis:{axis}"] + [f"{k}:{fields[k]}" for k in ("arm", "model") if fields.get(k)] + [f"cases:{cases}"]
    return {"axis": axis, "varies": varies, "holds": holds, "cases": cases, "scorers": scorers, **fields,
            "tags": tags, "description": f"{axis.upper()} axis. {description} Varies: {varies}. Holds: {holds}. Cases: {cases}. Scores: {scorers}."}


def classify_experiment(name: str) -> dict:
    """Experiment name -> factorial metadata (axis, varies, holds, cases, arm, loop, retriever, skill,
    model, ...) plus tags and a one-line description. Pure, testable."""
    if m := _ARM.match(name):
        arm, model, sha = m["arm"], _model_from_slug(m["model"]), m["sha"]
        short = _short(model)
        cases = "test-18" if short == "haiku" else "test-32"
        if sha == "3fcdae7" or short not in ("glm", "haiku"):
            return _factorial("model", varies="model", holds=f"system=D (agent, hybrid_rrf, skill on); cases={cases}", cases=cases,
                              scorers="obj/* (deterministic, MAUD gold spans)", arm="D", model=short, git_sha=sha, milestone="M6",
                              description=f"Arm D on {model}, the Pareto sweep run.")
        return _factorial("system", varies="system A->B->C->D, one config key per step", holds=f"model={short}; cases={cases}; index e2b4a2b97561",
                          cases=cases, scorers="obj/* (deterministic, MAUD gold spans)", arm=arm, model=short, git_sha=sha, milestone="M4/M4.1",
                          description=f"Arm {arm} = {ARM_SHAPE[arm][0]} + {ARM_SHAPE[arm][1]}, skill {ARM_SHAPE[arm][2]}, on {model}; {ARM_STEP[arm]}.")
    if m := _JUDGED.match(name):
        arm, model = m["arm"], _model_from_slug(m["model"])
        return _factorial("judge", varies="variant (arm@model) under the same three judges", holds="cases=judged-18; rubric cfda9f8cc401",
                          cases="judged-18", scorers="judge/* (mean of Mistral, NVIDIA, ByteDance)", arm=arm, model=_short(model), git_sha=m["sha"],
                          milestone="M5", description=f"Original M5 judged run of {arm}@{_short(model)}.")
    if m := _JUDGE.match(name):
        variant = m["variant"]; arm, short = variant.split("@", 1)
        return _factorial("judge", varies="variant (arm@model) under the same three judges and the same lawyer", holds="cases=judged-18; rubric cfda9f8cc401; blinded packets",
                          cases="judged-18", scorers="judge/* (mean of three judge families) + human/* (the lawyer, 24 packets)", arm=arm, model=short,
                          variant_id=variant, milestone="M5/M7b", description=f"{variant}: judge scores and the lawyer's scores on the same 18 rows.")
    if m := _PARETO.match(name):
        model = _model_from_slug(m["model"]); short = _short(model)
        cases = "test-18" if short == "haiku" else "test-32"
        return _factorial("model", varies="model (economics view: $/case, latency in row metadata)", holds=f"system=D; cases={cases}", cases=cases,
                          scorers="obj/* + $/case, wall_ms in metadata", arm="D", model=short, milestone="M6",
                          description=f"Arm D on {model}: the same run as D-{m['model']}, one row per case with cost and latency.")
    if m := _RAG.match(name):
        cfg = m["config"]; retriever = RAG_CONFIG.get(cfg, cfg)
        if cfg == "m7-synthetic":
            return _factorial("retrieval", varies="query distribution (58 canonical dev queries -> 106 LlamaIndex-generated ones)", holds="retriever=hybrid_rrf (the frozen winner)",
                              cases="synthetic-106", scorers="li/hit_rate, li/mrr", retriever=retriever, milestone="M7a",
                              description="Does the tournament winner hold up on questions it was not tuned on?")
        if cfg == "m7-li-crosscheck":
            return _factorial("retrieval", varies="scorer (LlamaIndex-native li/* vs our obj/* on the same hits)", holds="retriever=li_native_bm25; cases=dev-58",
                              cases="dev-58", scorers="li/* + obj/*", retriever=retriever, milestone="M7a",
                              description="Two independent scorers on one retriever; the three disagreements are real and explained.")
        return _factorial("retrieval", varies="retriever (dense / bm25 / hybrid_rrf / +rerank / fusion / +rerank)", holds="cases=dev-58; index e2b4a2b97561; k=5,10",
                          cases="dev-58", scorers="obj/hit@5, obj/hit@10, obj/mrr + li/hit_rate, li/mrr", retriever=retriever, milestone="M3/M7a",
                          description=f"Retriever {retriever} on the 58 dev queries.")
    if name == "deepeval-crosscheck":
        return _factorial("crosscheck", varies="evaluator framework (DeepEval vs our judges vs deterministic truth)", holds="cases=judged-18 x 6 variants (108 traces)",
                          cases="judged-18", scorers="deepeval/task_completion, tool_correctness, argument_correctness, step_efficiency", milestone="M7a",
                          description="DeepEval's independent read of the 108 judged traces; its evaluator model is a judge-trio member (contamination recorded).")
    if name == "m7b-hero-case":
        return _factorial("traces", varies="-", holds="one case, two variants", cases="hero-1", scorers="judge/<family> spans + judge/aggregate", milestone="M7b",
                          description="Hero case replay: three judge spans plus aggregate under scoring, A@haiku vs D@haiku, zero model calls.")
    if name == "m7-representative-traces":
        return _factorial("traces", varies="-", holds="six rule-chosen traces, one per failure shape", cases="representative-6", scorers="none (score-free replay)",
                          milestone="M7b", description="Six representative traces: clean success, retrieval rescue, defined-term cross-ref, inefficient trajectory, wrong answer, abstention.")
    if name.startswith("playground-arm-A-"):
        variant = name.removeprefix("playground-arm-A-")
        return _factorial("prompt", varies="arm-A system prompt (base / terse / cite-first / abstain-first)", holds="system=A (pipeline, dense top-5, no skill); model=glm; cases=judged-18; the exact passages arm A retrieved",
                          cases="judged-18", scorers="judge/* LLM scorers (evidence, professional, reasoning)", arm="A", model="glm", prompt_variant=variant, milestone="demo",
                          description=f"Arm A prompt variant '{variant}' over the 18 arm-A packets, judged by the LLM scorers.")
    return _factorial("other", varies="-", holds="-", cases="-", scorers="-", description="Unclassified experiment.")

# --- plumbing -------------------------------------------------------------------

class Api:
    def __init__(self, key: str):
        self.h = {"Authorization": f"Bearer {key}"}

    def _call(self, method: str, path: str, **kw) -> dict:
        """One request with backoff: 429 honours Retry-After (else 15 s), and the fetch
        endpoint's intermittent 400 ("memory access out of bounds") is retried briefly."""
        for attempt in range(6):
            r = requests.request(method, f"{API}/{path}", headers=self.h, timeout=120, **kw)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After") or 15))
                continue
            if r.status_code == 400 and "out of bounds" in r.text and attempt < 5:
                time.sleep(1.5 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        r.raise_for_status()
        return r.json()

    def get(self, path: str, params: dict | None = None) -> dict:
        return self._call("GET", path, params=params or {})

    def patch(self, path: str, body: dict) -> dict:
        return self._call("PATCH", path, json=body)

    def post(self, path: str, body: dict) -> dict:
        return self._call("POST", path, json=body)

    def resolve_project(self) -> tuple[str, str]:
        """(project_id, owner_user_id) for PROJECT by name; sets the module globals."""
        global PROJECT_ID, OWNER_USER_ID
        objs = self.get("project", {"project_name": PROJECT, "limit": 5}).get("objects", [])
        if not objs:
            raise RuntimeError(f"Braintrust project {PROJECT!r} not found in this org (key from {ENV_FILE})")
        PROJECT_ID, OWNER_USER_ID = objs[0]["id"], objs[0].get("user_id") or ""
        return PROJECT_ID, OWNER_USER_ID

    def experiments(self) -> list[dict]:
        out, params = [], {"project_id": PROJECT_ID, "limit": 200}
        while True:
            objs = self.get("experiment", params).get("objects", [])
            out += objs
            if len(objs) < 200:
                return out
            params["starting_after"] = objs[-1]["id"]

    def fetch_rows(self, experiment_id: str) -> list[dict]:
        """All spans of an experiment via the POST fetch form (cursor in the body)."""
        out, cursor = [], None
        while True:
            body = {"limit": 200}
            if cursor:
                body["cursor"] = cursor
            payload = self.post(f"experiment/{experiment_id}/fetch", body)
            events = payload.get("events", [])
            out += events
            cursor = payload.get("cursor")
            if not cursor or not events:
                return out


def _is_root(event: dict) -> bool:
    """Root spans: the fetch API's `is_root`, else no parents. (`span_id == root_span_id` is NOT
    reliable on fetched events and silently matched nothing.)"""
    return bool(event.get("is_root")) if event.get("is_root") is not None else not event.get("span_parents")


def _assert_scoreless(events: list[dict]) -> None:
    for e in events:
        if "scores" in e:
            raise RuntimeError("showroom refuses to write scores; a payload carried a 'scores' key")


def _ledger_path() -> Path:
    return LEDGER_PATH or org_report_path("braintrust_score_ledger.jsonl", override_env="BRAINTRUST_LEDGER_FILE")


def _manifest_path() -> Path:
    return MANIFEST_PATH or org_report_path("showroom_manifest.json")


def _ledger(experiment: str, key: str, n_scores: int = 0) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"experiment": experiment, "key": key, "n_scores": n_scores,
                             "ts": datetime.now(UTC).isoformat(), "source": "showroom"}) + "\n")


def _ledger_keys() -> set[tuple[str, str]]:
    path = _ledger_path()
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            seen.add((row["experiment"], row["key"]))
    return seen


def _case_info(case_id: str) -> dict:
    try:
        from dealpoint.eval.cases import find_case, resolve_question
        case = find_case(case_id)
        info = {"case_type": case.get("case_type"), "question_id": case.get("question_id")}
        with contextlib.suppress(Exception):     # reasoning type is decoration, never a failure
            info["reasoning_type"] = resolve_question(case).reasoning_type
        return info
    except Exception:  # noqa: BLE001 - unknown/redacted ids stay unannotated
        return {}


# --- steps ----------------------------------------------------------------------

def step_tag(api: Api, live: bool, manifest: dict) -> None:
    exps = api.experiments()
    plan = []
    for e in exps:
        c = classify_experiment(e["name"])
        plan.append({"id": e["id"], "name": e["name"], **{k: c.get(k) for k in ("axis", "arm", "model", "cases", "tags")}})
        if live:
            stale = {"stage", "family", "config", "variant_id", "git_sha", "arm", "model"}
            kept = {k: v for k, v in (e.get("metadata") or {}).items() if k not in stale}
            body = {"metadata": {**kept, **{k: v for k, v in c.items() if k not in ("tags", "description")}},
                    "tags": c["tags"], "description": c["description"]}
            api.patch(f"experiment/{e['id']}", body)
    manifest["experiments"] = plan
    by_axis: dict[str, int] = {}
    for p in plan:
        by_axis[p["axis"]] = by_axis.get(p["axis"], 0) + 1
    print(f"tag: {len(plan)} experiments by axis {by_axis}")


def step_rows(api: Api, live: bool, manifest: dict) -> None:
    seen = _ledger_keys()
    total = 0
    for e in manifest.get("experiments") or [{"id": x["id"], "name": x["name"]} for x in api.experiments()]:
        c = classify_experiment(e["name"])
        if (e["name"], "rows-metadata") in seen:
            continue
        rows = api.fetch_rows(e["id"]) if live else []
        events = []
        for r in rows:
            if not _is_root(r):
                continue
            case_id = r.get("input") if isinstance(r.get("input"), str) else (r.get("metadata") or {}).get("case_id")
            meta = {k: v for k, v in c.items() if k in ("axis", "arm", "loop", "retriever", "skill", "model", "variant_id", "cases")}
            if isinstance(case_id, str):
                meta["case_id"] = case_id
                meta.update(_case_info(case_id))
            events.append({"id": r["id"], "metadata": meta, "_is_merge": True, "_merge_paths": [["metadata"]]})
        _assert_scoreless(events)
        if live and events:
            for i in range(0, len(events), 100):
                api.post(f"experiment/{e['id']}/insert", {"events": events[i:i + 100]})
            _ledger(e["name"], "rows-metadata")
        total += len(events)
        print(f"rows: {e['name']}: {len(events)} root rows {'merged' if live else 'would merge'}")
        if live:
            time.sleep(0.6)                 # stay under the API rate limit across 42 experiments
    manifest["rows_merged"] = total


SYSTEM_LABEL = {"A": "A: pipeline + dense", "B": "B: agent + dense", "C": "C: agent + hybrid", "D": "D: agent + hybrid + skill"}
JUDGE_DIMS = ("reasoning", "evidence", "trajectory", "professional")


def _rescale(v) -> float | None:
    """Rubric 1..5 -> 0..1, the same map the LLM scorers' choice_scores use."""
    return None if v is None else (float(v) - 1.0) / 4.0


def _flag(v) -> int | None:
    return None if v is None else int(bool(v))


CANONICAL_CATEGORIES = ("judged", "agent")


def metadata_mirror(row: dict, *, case: dict | None = None, judge_dims: dict | None = None, human: dict | None = None,
                    deepeval: dict | None = None, per_judge: dict | None = None, category: str | None = None) -> dict:
    """Every number the dashboard needs, as log METADATA (free) rather than scores (metered), with one
    label vocabulary across all log families (short model names, self-describing system labels).
    The same values live as real scores on the experiments; this is the mirror the Monitor page can see."""
    scores = row.get("scores") or {}
    rec = row.get("record") or {}
    status = rec.get("status") or row.get("status")
    arm = row.get("arm") or rec.get("arm")
    model = row.get("model") or rec.get("model") or ""
    short = SHORT_MODEL.get(model, model)
    ga = scores.get("grounded_accuracy")
    out = {
        "system_label": SYSTEM_LABEL.get(arm, arm),  # type: ignore[call-overload,arg-type]
        "model_label": short, "variant_label": f"{arm}@{short}",
        "status": status, "case_set": (case or {}).get("case_set") or row.get("case_set"), "question_id": (case or {}).get("question_id") or row.get("question_id"),
        "ga_scored": None if ga is None else int(bool(ga)), "ga_all": int(bool(ga)),
        "answer_correct_all": int(bool(scores.get("answer_correct"))),
        "cap_hit": _flag(scores.get("cap_hit") if scores.get("cap_hit") is not None else status == "CAP_HIT"),
        "execution_failed": _flag(scores.get("execution_failed") if scores.get("execution_failed") is not None else status == "EXECUTION_FAILED"),
        "abstained": int(status == "ABSTAINED"), "fabrication": _flag(scores.get("fabrication")),
        "abstain_correct": _flag(scores.get("abstain_correct")) if ((case or {}).get("case_set") or row.get("case_set")) == "counterfactual" else None,
        "usd": scores.get("usd") if scores.get("usd") is not None else row.get("usd"),
        "wall_s": (scores.get("wall_ms") or rec.get("wall_ms") or 0) / 1000.0 or None,
        "tool_calls": scores.get("tool_calls") if scores.get("tool_calls") is not None else rec.get("tool_calls"),
        "has_human": int(bool(human)),
    }
    # Fair comparison fields. `correct_all` credits a correct abstention on a counterfactual case (where
    # there is nothing to ground) as a correct outcome, so an "accuracy over all cases" that includes
    # counterfactuals is not rigged against them. `comparable` marks the canonical sweep and judged
    # traces of the five slate models; representative picks, live replays and partial runs (a model that
    # never completed its sweep) are excluded from any chart or Pattern that ranks variants.
    # `case_pool` names the case set the trace belongs to, so a comparison stays within one pool.
    case_set = out["case_set"]
    out["correct_all"] = int(bool(ga) or (case_set == "counterfactual" and bool(scores.get("abstain_correct"))))
    # The headline scoring system. Failures are not equal in a legal tool: an answer that is wrong (or any
    # answer at all when the agreement does not address the question) MISLEADS; an abstention, a cap-hit or
    # an execution failure is a SILENT failure that costs a lookup, not a lawsuit. net_accuracy per row is
    # +1 correct, -1 misleading, 0 silent; its mean over a case pool is "correct minus misleading".
    out["misleading"] = int(status == "ANSWERED" and not out["correct_all"])
    out["silent_failure"] = int(not out["correct_all"] and not out["misleading"])
    out["net_accuracy"] = out["correct_all"] - out["misleading"]
    # The trust pair. safe = did not mislead (correct or silent); it is inflated by silence, so it is always
    # charted next to precision-when-answering (sum(correct_answered)/sum(answered)), which punishes silence.
    # verbatim: when it answered, was the quoted evidence a verbatim span of the agreement.
    out["safe"] = 1 - out["misleading"]
    out["answered"] = int(status == "ANSWERED")
    out["correct_answered"] = int(out["answered"] and out["correct_all"])
    out["cite_verbatim_answered"] = int(out["answered"] and bool(scores.get("citation_verbatim"))) if out["answered"] else None
    out["cite_gold_answered"] = int(out["answered"] and bool(scores.get("citation_gold_overlap"))) if out["answered"] else None
    out["comparable"] = int(category in CANONICAL_CATEGORIES and short in SHORT_MODEL.values())
    out["case_pool"] = {"judged": "judged-18 (the same 18 cases for every variant)", "agent": "test-32 (the frozen 32-case subset)"}.get(category or "", "other")
    for d in JUDGE_DIMS:
        out[f"judge_{d}"] = _rescale((judge_dims or {}).get(d))
        out[f"human_{d}"] = _rescale((human or {}).get(d))
        if out[f"judge_{d}"] is not None and out[f"human_{d}"] is not None:
            # panel closeness: 1 - |panel mean - lawyer| on the 0..1 scale; 1.0 means identical
            out[f"panel_{d}_closeness"] = 1.0 - abs(out[f"judge_{d}"] - out[f"human_{d}"])
        for family, v in (per_judge or {}).items():
            out[f"judge_{family.lower()}_{d}"] = _rescale(v.get(d))
            if human and v.get(d) is not None and human.get(d) is not None:
                out[f"judge_{family.lower()}_{d}_abs_err"] = abs(_rescale(v[d]) - _rescale(human[d]))  # type: ignore[operator]
                out[f"judge_{family.lower()}_{d}_bias"] = _rescale(v[d]) - _rescale(human[d])     # + = judge runs high  # type: ignore[operator]
                out[f"judge_{family.lower()}_{d}_closeness"] = 1.0 - out[f"judge_{family.lower()}_{d}_abs_err"]   # 1.0 = identical
    # Per judge family: the realized cost of its call on this packet, the mean distance from the lawyer
    # over the four dimensions, and agreement per dollar ((1 - distance) / usd), so "which judge" can be
    # answered per dimension, overall, and with cost in the picture.
    for family, v in (per_judge or {}).items():
        f = family.lower()
        out[f"judge_{f}_usd"] = v.get("usd")
        errs = [out[k] for k in (f"judge_{f}_{d}_abs_err" for d in JUDGE_DIMS) if out.get(k) is not None]
        if errs:
            out[f"judge_{f}_mean_abs_err"] = sum(errs) / len(errs)
            out[f"judge_{f}_agreement_per_dollar"] = (1 - out[f"judge_{f}_mean_abs_err"]) / v["usd"] if v.get("usd") else None
    for k in ("task_completion", "tool_correctness", "argument_correctness", "step_efficiency"):
        v = ((deepeval or {}).get(k) or {}).get("score") if deepeval else None
        out[f"deepeval_{k}"] = v
    tc = out["deepeval_task_completion"]
    out["deepeval_agrees_with_truth"] = None if (tc is None or ga is None) else int((tc >= 0.5) == bool(ga))
    jr, hr = out.get("judge_reasoning"), out.get("human_reasoning")
    out["deepeval_agrees_with_judges"] = None if (tc is None or jr is None) else int((tc >= 0.5) == (jr >= 0.5))
    out["deepeval_agrees_with_lawyer"] = None if (tc is None or hr is None) else int((tc >= 0.5) == (hr >= 0.5))
    return out


def _mirror_context() -> dict:
    """Judge means per packet, the lawyer's scores per packet, DeepEval per (case, variant): loaded once."""
    from dealpoint.eval.braintrust_cockpit import _load_jsonl
    from dealpoint.eval.braintrust_sync import (
        _deepeval_scores_by_trace,
        _mean_judge_dims_for_packet,
        _packet_id_for,
    )

    judge_rows = _load_jsonl("data/eval/judge_scores.jsonl")
    human_path = Path("data/eval/calibration/human_scores.jsonl")
    human = {h["packet_id"]: h for h in _load_jsonl(human_path)} if human_path.exists() else {}
    return {"judge_rows": judge_rows, "human": human, "deepeval": _deepeval_scores_by_trace(),
            "packet_id_for": _packet_id_for, "judge_dims_for": _mean_judge_dims_for_packet}


def mirror_for(case_id: str, variant_id: str, row: dict, ctx: dict, category: str | None = None) -> dict:
    """`metadata_mirror` for one (case, variant) under either id convention (`D@glm` or `D@z-ai/glm-5.3-flash`)."""
    from dealpoint.eval.braintrust_sync import _judged_variant_id_for
    from dealpoint.eval.cases import find_case

    arm, _, model = variant_id.partition("@")
    short_variant = variant_id if model in SHORT_MODEL.values() else (_judged_variant_id_for(arm, model) or variant_id)
    try:
        case = find_case(case_id)
    except KeyError:
        case = {}
    pid = ctx["packet_id_for"](case_id, short_variant)
    judge_dims = ctx["judge_dims_for"](ctx["judge_rows"], pid) or None
    per_judge = {r["judge_family"]: r for r in ctx["judge_rows"] if r.get("packet_id") == pid and r.get("ok", True)}
    human = ctx["human"].get(pid)
    deepeval = ctx["deepeval"].get((case_id, short_variant))
    row = {**row, "arm": row.get("arm") or arm, "model": row.get("model") or model}
    return metadata_mirror(row, case=case, judge_dims=judge_dims, human=human, deepeval=deepeval, per_judge=per_judge or None, category=category)


def stored_row_index() -> dict[tuple[str, str], dict]:
    """(case_id, variant_id) -> stored result row, under BOTH id conventions, from the judged subset,
    the four-arm/Pareto sweeps and the representative selections."""
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _load_jsonl, _variant_results
    from dealpoint.eval.braintrust_sync import representative_cases

    idx: dict[tuple[str, str], dict] = {}
    subset = _judged_subset()
    for v in subset.get("variants", []):
        for case_id, row in _variant_results(subset, v["variant_id"]).items():
            idx[(case_id, v["variant_id"])] = row
            idx[(case_id, f"{v['arm']}@{v['model']}")] = row
    for p in agent_run_log_plan(set()):
        idx.setdefault((p["case_id"], p["variant_id"]), p["row"])
    for sel in representative_cases().get("selections", []):
        if sel.get("results_path"):
            for r in _load_jsonl(sel["results_path"]):
                if r.get("case_id") == sel["case_id"]:
                    idx.setdefault((sel["case_id"], sel["variant_id"]), r)
    return idx


def llamaindex_per_case() -> dict[tuple[str, str], dict]:
    """(case_id, retriever) -> LlamaIndex's own hit_rate and mrr for that query, from data/reports/li_rag_eval.json."""
    path = Path("data/reports/li_rag_eval.json")
    if not path.exists():
        return {}
    li = json.loads(path.read_text(encoding="utf-8")).get("retrievers", {})
    out: dict[tuple[str, str], dict] = {}
    for config, block in li.items():
        for r in (block.get("li") or {}).get("per_case", []):
            out[(r["case_id"], config)] = {"li_hit_rate": r.get("hit_rate"), "li_mrr": r.get("mrr")}
    return out


def retrieval_log_rows() -> list[dict]:
    """The M3 tournament, one score-free log per (dev case, retriever config) on the canonical query:
    gold-span rank as metadata (`hit_at_5`, `hit_at_10`, `mrr`), so the dashboard can rank retrievers."""
    from dealpoint.eval.cases import find_case, resolve_question

    t = json.loads(Path("data/reports/tournament.json").read_text(encoding="utf-8"))
    li = llamaindex_per_case()
    per_case = [r for r in t.get("per_case", []) if r.get("query_type") == "canonical"]
    # LlamaIndex's genuinely native config (li_native_bm25) has no tournament row: its rows come from the
    # LlamaIndex report alone, with our obj/ scorer's verdict absent (that is the cross-check's point).
    seen = {(r["case_id"], r["config"]) for r in per_case}
    for case_id, config in li:
        if config == "li_native_bm25" and (case_id, config) not in seen:
            per_case.append({"case_id": case_id, "config": config, "query_type": "canonical", "first_hit_rank": None, "agreement_id": case_id.split("__")[0],
                             "question_id": case_id.split("__")[-1], "li_only": True})
    out: list[dict] = []
    for r in per_case:
        rank = r.get("first_hit_rank")
        try:
            q = resolve_question(find_case(r["case_id"]))
            query, gloss = q.canonical_query, q.gloss
        except KeyError:
            query, gloss = r["case_id"], ""
        li_only = bool(r.get("li_only"))
        meta = {"category": "retrieval", "retriever": r["config"], "case_id": r["case_id"], "question_id": r.get("question_id"),
                "agreement_id": r.get("agreement_id"), "gloss": gloss, "first_hit_rank": rank, "stages": r.get("stages"),
                "scored_by_ours": int(not li_only), **li.get((r["case_id"], r["config"]), {})}
        if not li_only:
            meta.update({"hit_at_5": int(bool(rank and rank <= 5)), "hit_at_10": int(bool(rank and rank <= 10)), "mrr": (1.0 / rank) if rank else 0.0})
        output = f"LlamaIndex-native config: li/hit_rate {meta.get('li_hit_rate')}" if li_only else (f"first gold-span hit at rank {rank}" if rank else "no gold-span hit in the top 20")
        out.append({"case_id": r["case_id"], "retriever": r["config"], "input": query, "output": output, "metadata": meta})
    return out


def prompt_variant_log_rows(events_by_variant: dict[str, list[dict]]) -> list[dict]:
    """Playground experiments (`playground-arm-A-<variant>` fetch events) -> one score-free log per
    (variant, case) with the judges' scores as metadata. Pure over fetched events, testable."""
    out: list[dict] = []
    for variant, events in events_by_variant.items():
        roots = {e["root_span_id"]: e for e in events if _is_root(e)}
        judged: dict[str, dict] = {}
        for e in events:
            if (e.get("span_attributes") or {}).get("type") == "score" and e.get("scores"):
                for name, val in e["scores"].items():
                    judged.setdefault(e["root_span_id"], {})[name.replace("Judge: ", "judge_")] = val
        for rid, root in roots.items():
            meta = root.get("metadata") or {}
            case_id = meta.get("case_id") or meta.get("id")
            answer = root.get("output")
            if isinstance(answer, dict):
                answer = answer.get("answer") or json.dumps(answer)[:200]
            out.append({"variant": variant, "case_id": case_id, "input": str(root.get("input") or "")[:2000], "output": str(answer)[:500] if answer is not None else "(no output)",
                        "metadata": {"category": "prompt-variant", "prompt_variant": variant, "case_id": case_id, "question_id": meta.get("question_id"),
                                     "gold_answer": meta.get("gold_answer"), "case_set": meta.get("case_set"), "system_label": SYSTEM_LABEL["A"], "model_label": "glm",
                                     **judged.get(rid, {})}})
    return out


def agent_run_log_plan(already_planned: set[tuple[str, str]]) -> list[dict]:
    """Every stored agent run (the four-arm and Pareto sweeps, one row per case) as a log
    tree, category `agent`, minus the (case, variant) pairs the judged/representative plan
    already covers under their short variant ids (`D@glm`) or full ones (`D@z-ai/glm-5.3-flash`).
    Logs cost processed data only, never scores; the fuller the Logs tab, the more Topics,
    Patterns and online scoring have to work with."""
    from dealpoint.eval.braintrust_sync import _agent_experiment_plans, _judged_variant_id_for

    out: list[dict] = []
    for plan in _agent_experiment_plans():
        arm, model = plan["metadata"]["arm"], plan["metadata"]["model"]
        variant_id = f"{arm}@{model}"
        short = _judged_variant_id_for(arm, model)
        for row in plan["rows"]:
            case_id = row.get("case_id")
            if not case_id or (case_id, variant_id) in already_planned or (short and (case_id, short) in already_planned):
                continue
            out.append({"case_id": case_id, "variant_id": variant_id, "category": "agent", "row": row})
    return out


def step_logs(api: Api, live: bool, manifest: dict) -> None:
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _load_jsonl, _variant_results
    from dealpoint.eval.braintrust_sync import representative_cases

    subset = _judged_subset()
    plan: list[dict] = []
    for variant in subset.get("variants", []):
        rows = _variant_results(subset, variant["variant_id"])
        for case_id in subset.get("case_ids", list(rows)):
            plan.append({"case_id": case_id, "variant_id": variant["variant_id"], "category": "judged", "row": rows.get(case_id)})
    for sel in representative_cases().get("selections", []):
        rows = {r.get("case_id"): r for r in _load_jsonl(sel["results_path"])} if sel.get("results_path") else {}
        plan.append({"case_id": sel["case_id"], "variant_id": sel["variant_id"], "category": sel["category"], "row": rows.get(sel["case_id"])})
    plan += agent_run_log_plan({(p["case_id"], p["variant_id"]) for p in plan})
    seen = _ledger_keys()
    todo = [p for p in plan if ("logs", f"{p['case_id']}:{p['variant_id']}:{p['category']}") not in seen]
    print(f"logs: {len(plan)} trace trees planned, {len(todo)} not yet in the ledger, 0 scores")
    manifest["logs"] = {"planned": len(plan), "written_this_run": 0, "roots": []}
    if not live or not todo:
        return
    import braintrust
    logger = braintrust.init_logger(project=PROJECT, set_current=True)
    for p in todo:
        _log_one_tree(logger, p, manifest)
    logger.flush()


def _log_one_tree(logger, p: dict, manifest: dict) -> str | None:
    """Replay one stored result row into Logs as a full span tree with zero scores; returns the root id."""
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.braintrust_sync import _emit_span_tree, log_hierarchy
    from dealpoint.eval.cases import find_case, resolve_document_id

    row = p["row"] or {"case_id": p["case_id"], "scores": {}, "record": {"trajectory": []}}
    try:
        case = find_case(p["case_id"]); doc = load_document(resolve_document_id(case))
    except (KeyError, FileNotFoundError):
        case, doc = {}, None
    hierarchy = log_hierarchy(row, case, doc, judge_dims=None)
    arm, _, model = p["variant_id"].partition("@")
    meta = {"case_id": p["case_id"], "variant_id": p["variant_id"], "arm": arm, "model": model, "category": p["category"],
            "status": (row.get("record") or {}).get("status") or row.get("status"),
            "obj_grounded_accuracy": (row.get("scores") or {}).get("grounded_accuracy"), **_case_info(p["case_id"]),
            **mirror_for(p["case_id"], p["variant_id"], row, p.get("ctx") or _mirror_context(), category=p["category"])}
    root = logger.start_span(name=f"{p['case_id']} | {p['variant_id']}")
    root.log(input=(case or {}).get("question_text") or p["case_id"], metadata=meta)
    n = 0
    for child in hierarchy.get("children", []):
        n += _emit_span_tree(logger, child, parent_span=root, allowed_score_names=set())
    if n:
        raise RuntimeError(f"replay emitted {n} scores; expected zero")
    finding = row.get("finding") or (row.get("record") or {}).get("finding")
    root.log(output=finding if finding is not None else "(no finding)")
    root.end()
    root_id = getattr(root, "id", None)
    manifest.setdefault("logs", {"planned": 0, "written_this_run": 0, "roots": []})
    manifest["logs"]["roots"].append({"case_id": p["case_id"], "variant_id": p["variant_id"], "category": p["category"], "root_span_id": root_id})
    _ledger("logs", f"{p['case_id']}:{p['variant_id']}:{p['category']}")
    manifest["logs"]["written_this_run"] += 1
    return root_id


def replay_plan_entry(case_id: str, variant_id: str) -> dict:
    """The stored row for `case_id` under a judged short id (`D@glm`) or a full one
    (`D@z-ai/glm-5.3-flash`), as a log-plan entry whose category carries a timestamp, so the
    ledger never treats it as already logged: this is the demo's "a new trace lands" action."""
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _variant_results

    subset = _judged_subset()
    row = _variant_results(subset, variant_id).get(case_id) if any(v["variant_id"] == variant_id for v in subset.get("variants", [])) else None
    if row is None:
        row = next((p["row"] for p in agent_run_log_plan(set()) if p["case_id"] == case_id and p["variant_id"] == variant_id), None)
    if row is None:
        raise SystemExit(f"no stored result row for {case_id} {variant_id}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return {"case_id": case_id, "variant_id": variant_id, "category": f"live-replay-{stamp}", "row": row}


def step_replay(api: Api, live: bool, manifest: dict, target: str) -> None:
    case_id, _, variant_id = target.partition(":")
    p = replay_plan_entry(case_id, variant_id)
    print(f"replay: {case_id} as {variant_id} -> Logs, category {p['category']}, 0 scores (online scoring picks it up)")
    if not live:
        return
    import braintrust
    logger = braintrust.init_logger(project=PROJECT, set_current=True)
    root_id = _log_one_tree(logger, p, manifest)
    logger.flush()
    print(f"  logged root span {root_id}")


def _all_root_logs(api: Api) -> list[dict]:
    out, cursor = [], None
    while True:
        body = {"limit": 500, **({"cursor": cursor} if cursor else {})}
        payload = api.post(f"project_logs/{PROJECT_ID}/fetch", body)
        events = payload.get("events", [])
        out += [e for e in events if _is_root(e)]
        cursor = payload.get("cursor")
        if not cursor or not events:
            return out


def step_mirror(api: Api, live: bool, manifest: dict) -> None:
    """Merge the metadata mirror onto every existing agent-trace root log (idempotent; 0 scores)."""
    idx = stored_row_index()
    print(f"mirror: metadata mirror (labels, obj/judge/human/DeepEval numbers, cost, latency) onto every root log; {len(idx)} stored rows indexed")
    manifest["mirror"] = {"rows_indexed": len(idx), "logs_updated": 0, "logs_unmatched": 0}
    if not live:
        return
    ctx = _mirror_context()
    roots = _all_root_logs(api)
    events, unmatched = [], 0
    li = llamaindex_per_case()
    for e in roots:
        m = e.get("metadata") or {}
        if m.get("category") == "retrieval":
            extra = li.get((m.get("case_id"), m.get("retriever")))  # type: ignore[arg-type]
            if extra:
                events.append({"id": e["id"], "metadata": {**extra, "scored_by_ours": int(m.get("retriever") != "li_native_bm25")}, "_is_merge": True})
            continue
        if m.get("category") == "prompt-variant" or not m.get("case_id") or not m.get("variant_id"):
            continue
        row = idx.get((m["case_id"], m["variant_id"]))
        if row is None:
            unmatched += 1
            continue
        events.append({"id": e["id"], "metadata": mirror_for(m["case_id"], m["variant_id"], row, ctx, category=m.get("category")), "_is_merge": True})
    _assert_scoreless(events)
    for i in range(0, len(events), 100):
        api.post(f"project_logs/{PROJECT_ID}/insert", {"events": events[i:i + 100]})
    manifest["mirror"].update({"logs_updated": len(events), "logs_unmatched": unmatched})
    _ledger("logs", f"mirror:{len(events)}")
    print(f"  merged onto {len(events)} logs ({unmatched} unmatched)")


def step_raglogs(api: Api, live: bool, manifest: dict) -> None:
    rows = retrieval_log_rows()
    seen = _ledger_keys()
    todo = [r for r in rows if ("logs", f"retrieval:{r['case_id']}:{r['retriever']}") not in seen]
    print(f"raglogs: {len(rows)} retrieval logs (58 dev queries x 6 tournament retrievers + LlamaIndex's native bm25; gold-span rank and LlamaIndex's own hit/mrr as metadata), {len(todo)} not yet in the ledger, 0 scores")
    manifest["raglogs"] = {"planned": len(rows), "written_this_run": 0}
    if not live or not todo:
        return
    import braintrust
    logger = braintrust.init_logger(project=PROJECT, set_current=True)
    for r in todo:
        span = logger.start_span(name=f"{r['case_id']} | retrieval:{r['retriever']}")
        span.log(input=r["input"], output=r["output"], metadata=r["metadata"])
        span.end()
        _ledger("logs", f"retrieval:{r['case_id']}:{r['retriever']}")
        manifest["raglogs"]["written_this_run"] += 1
    logger.flush()


def step_promptlogs(api: Api, live: bool, manifest: dict) -> None:
    """The Playground pre-run experiments mirrored into Logs, one per (variant, case), judges as metadata."""
    variants = [pr["slug"].removeprefix("arm-a-prompt-") for pr in playground_prompts()]
    print(f"promptlogs: playground-arm-A-* experiments -> logs with judge scores as metadata ({', '.join(variants)}), 0 scores")
    manifest["promptlogs"] = {"variants": variants, "written_this_run": 0}
    if not live:
        return
    events_by_variant: dict[str, list[dict]] = {}
    for v in variants:
        exp = api.get("experiment", {"project_id": PROJECT_ID, "experiment_name": f"playground-arm-A-{v}"}).get("objects", [])
        if exp:
            events_by_variant[v] = api.fetch_rows(exp[0]["id"])
    rows = prompt_variant_log_rows(events_by_variant)
    seen = _ledger_keys()
    todo = [r for r in rows if r["case_id"] and ("logs", f"prompt:{r['variant']}:{r['case_id']}") not in seen]
    print(f"  {len(rows)} rows fetched, {len(todo)} to log")
    if not todo:
        return
    import braintrust
    logger = braintrust.init_logger(project=PROJECT, set_current=True)
    for r in todo:
        span = logger.start_span(name=f"{r['case_id']} | prompt:{r['variant']}")
        span.log(input=r["input"], output=r["output"], metadata=r["metadata"])
        span.end()
        _ledger("logs", f"prompt:{r['variant']}:{r['case_id']}")
        manifest["promptlogs"]["written_this_run"] += 1
    logger.flush()


def step_review(api: Api, live: bool, manifest: dict) -> None:
    from dealpoint.eval.braintrust_sync import review_set
    items = review_set()
    exps = {e["name"]: e["id"] for e in api.experiments()}
    flagged = []
    for it in items:
        exp_name = f"judge-{it['variant_id']}"
        if exp_name not in exps:
            continue
        flagged.append({"case_id": it["case_id"], "variant_id": it["variant_id"], "experiment": exp_name})
    print(f"review: {len(flagged)} of {len(items)} review-set rows resolve to judge experiments")
    manifest["review_flagged"] = flagged
    if not live:
        return
    by_exp: dict[str, list[str]] = {}
    for f in flagged:
        by_exp.setdefault(f["experiment"], []).append(f["case_id"])
    for exp_name, case_ids in by_exp.items():
        rows = {r.get("input"): r["id"] for r in api.fetch_rows(exps[exp_name]) if _is_root(r)}
        events = [{"id": rows[c], "metadata": {"~__bt_review_lists": {"__bt_default_review_list": {"status": "PENDING"}},
                                                "~__bt_assignments": [OWNER_USER_ID]},
                   "_is_merge": True, "_merge_paths": [["metadata", "~__bt_review_lists"], ["metadata", "~__bt_assignments"]]}
                  for c in case_ids if c in rows]
        _assert_scoreless(events)
        if events:
            api.post(f"experiment/{exps[exp_name]}/insert", {"events": events})
            _ledger(exp_name, "review-flags")


def arm_parameter_sets() -> list[dict]:
    from dealpoint.eval.braintrust_sync import parameters_schema
    s = parameters_schema()
    retr = s.get("arm_retrievers", {"A": "dense", "B": "dense", "C": "hybrid_rrf", "D": "hybrid_rrf"})
    # dealpoint.config.ARMS is the truth: each arm differs from its predecessor by exactly one key
    # (A->B the loop, B->C the retriever, C->D the skill), so these objects must say the same.
    from dealpoint.config import ARM_ORDER, ARMS
    descriptions = {
        "A": "Pipeline, dense retrieval: top-k dense hits into one call, one answer. The control.",
        "B": "Agent loop on dense retrieval: tools (search_agreement, get_section, lookup_defined_term), no skill. Isolates the loop.",
        "C": "Agent loop on the tournament-winning hybrid retrieval (dense + BM25, RRF). Isolates the retriever.",
        "D": "Agent loop, hybrid retrieval, skill injected into the system prompt. Isolates the skill.",
    }
    return [
        {"arm": arm, "retriever": retr.get(arm, ARMS[arm]["retriever"]["name"]), "top_k": s.get("top_k", 5),
         "max_tool_calls": s.get("max_tool_calls", 8) if ARMS[arm]["loop"] == "agent" else 0,
         "agent_loop": ARMS[arm]["loop"] == "agent", "skill": bool(ARMS[arm]["skill"]),
         "description": descriptions[arm]}
        for arm in ARM_ORDER
    ]


def step_params(api: Api, live: bool, manifest: dict) -> None:
    sets = arm_parameter_sets()
    print(f"params: {len(sets)} arm parameter sets: " + ", ".join(f"arm-{s['arm']}" for s in sets))
    manifest["parameters"] = [f"arm-{s['arm']}-config" for s in sets]
    if not live:
        return
    import braintrust
    from pydantic import create_model
    project = braintrust.projects.create(name=PROJECT)
    Model = create_model("ArmConfig", retriever=(str, "dense"), top_k=(int, 5), max_tool_calls=(int, 0), agent_loop=(bool, False), skill=(bool, False))
    for s in sets:
        project.parameters.create(name=f"arm-{s['arm']}-config", slug=f"arm-{s['arm'].lower()}-config", schema={"parameters": Model},
                                  if_exists="replace", metadata={k: v for k, v in s.items()})
    project.publish()
    _ledger("parameters", "arm-configs")


def arm_a_prompt_variants() -> list[dict]:
    from dealpoint.agent.prompts import system_prompt
    base = system_prompt()
    return [
        {"slug": "arm-a-prompt-terse", "name": "Arm A prompt: terse",
         "content": base + "\n\nAnswer in at most two sentences. Quote the operative contract language once, then stop."},
        {"slug": "arm-a-prompt-cite-first", "name": "Arm A prompt: cite first",
         "content": base + "\n\nBefore answering, quote the exact clause you rely on, verbatim. Only then state the answer, and only if the quote establishes it."},
        {"slug": "arm-a-prompt-abstain-first", "name": "Arm A prompt: abstain first",
         "content": base + "\n\nIf the retrieved text does not contain the defined term or clause the question turns on, say so and abstain. Never infer an absence from an introduction to a list."},
    ]


def step_prompts(api: Api, live: bool, manifest: dict) -> None:
    variants = arm_a_prompt_variants()
    print("prompts: " + ", ".join(v["slug"] for v in variants) + f" (model {WORKHORSE_MODEL} via OpenRouter)")
    manifest["prompt_variants"] = [v["slug"] for v in variants]
    if not live:
        return
    import braintrust
    project = braintrust.projects.create(name=PROJECT)
    for v in variants:
        project.prompts.create(name=v["name"], slug=v["slug"], prompt=v["content"], model=WORKHORSE_MODEL, if_exists="replace",
                               metadata={"variant_of": "agent-base-system-prompt", "purpose": "playground A/B/C"})
    project.publish()
    _ledger("prompts", "arm-a-variants")


PLAYGROUND_DATASET = "maud-dealpoint-playground-armA"
PLAYGROUND_USER_TURN = "{{input}}\n\nGive your final answer based only on the passages above."
PLAYGROUND_JUDGES = ("evidence", "professional", "reasoning")     # trajectory is meaningless for a single-shot prompt


def playground_rows() -> list[dict]:
    """The 18 judged cases as arm-A packets: the question block and the five dense top-5 passages arm A
    actually retrieved (from the stored A@haiku trajectory's char ranges), rendered exactly as the pipeline
    rendered them. A Playground run over these rows is an arm-A prompt A/B/C with everything but the
    system prompt held fixed. `expected` carries the gold answer and gold span for the evidence judge."""
    from types import SimpleNamespace

    from dealpoint.agent.pipeline import _render_context
    from dealpoint.agent.prompts import out_of_scope_question_block, question_spec_block
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _variant_results
    from dealpoint.eval.cases import find_case, resolve_document_id, resolve_question

    subset = _judged_subset()
    rows = _variant_results(subset, "A@haiku")
    out: list[dict] = []
    for case_id in subset.get("case_ids", []):
        row = rows.get(case_id)
        if not row:
            continue
        case = find_case(case_id); doc = load_document(resolve_document_id(case)); q = resolve_question(case)
        step = ((row.get("record") or {}).get("trajectory") or [{}])[0]
        chunks = [SimpleNamespace(text=doc.text[a:b], section_ref=None, start=a, end=b) for a, b in step.get("char_ranges") or []]
        block = question_spec_block(q) if q.options else out_of_scope_question_block(q.gloss)
        spans = [doc.text[sp["start"]:sp["end"]] for sp in case.get("gold_spans") or []]
        expected = case.get("gold_answer", "ABSTAIN") + (("\n\nGold span: " + " ... ".join(spans)) if spans else "")
        out.append({"id": case_id, "input": block + "\n\nRetrieved passages from this agreement:\n" + _render_context(doc, chunks),
                    "expected": expected,
                    "metadata": {"case_id": case_id, "question_id": q.id, "gloss": q.gloss, "gold_answer": case.get("gold_answer"),
                                 "case_set": case.get("case_set"), "reasoning_type": q.reasoning_type, "n_passages": len(chunks), "arm": "A"}})
    return out


def playground_prompts() -> list[dict]:
    """Base arm-A system prompt plus the three variants, as chat prompts with a `{{input}}` user turn."""
    from dealpoint.agent.prompts import system_prompt

    base = {"slug": "arm-a-prompt-base", "name": "Arm A prompt: base", "content": system_prompt()}
    return [{**v, "messages": [{"role": "system", "content": v["content"]}, {"role": "user", "content": PLAYGROUND_USER_TURN}]}
            for v in [base, *arm_a_prompt_variants()]]


JUDGE_PACKETS_DATASET = "maud-dealpoint-judge-packets"
JUDGE_PANEL_PROMPT_SLUG = "judge-panel-rubric"
JUDGE_PANEL_MODELS = {"Mistral": "mistralai/mistral-small-3.2-24b-instruct", "NVIDIA": "nvidia/nemotron-3-super-120b-a12b", "ByteDance": "bytedance-seed/seed-2.0-mini"}


def judge_packet_rows() -> list[dict]:
    """The 24 lawyer-scored blinded packets exactly as the M5 judges saw them (`render_packet_text`), with the
    lawyer's four scores as `expected` and each judge family's scores in metadata. A Playground of the
    rubric prompt over these rows, one column per judge model, is "watch the judges argue with the lawyer"."""
    from dealpoint.eval.blinding import render_packet_text
    from dealpoint.eval.braintrust_cockpit import _load_jsonl

    packets = {p["packet_id"]: p for p in _load_jsonl("data/eval/calibration/packets.jsonl")}
    human = {h["packet_id"]: h for h in _load_jsonl("data/eval/calibration/human_scores.jsonl")}
    key = json.loads(Path("data/eval/calibration/variant_key.json").read_text(encoding="utf-8"))
    judges = _load_jsonl("data/eval/judge_scores.jsonl")
    out: list[dict] = []
    for pid, h in sorted(human.items()):
        packet = packets.get(pid)
        if not packet:
            continue
        meta = key.get(pid, {})
        per_judge = {r["judge_family"].lower(): {d: r.get(d) for d in JUDGE_DIMS} for r in judges if r.get("packet_id") == pid and r.get("ok", True)}
        expected = {d: h.get(d) for d in JUDGE_DIMS}
        out.append({"id": pid, "input": render_packet_text(packet),
                    "expected": json.dumps(expected),
                    "metadata": {"packet_id": pid, "case_id": meta.get("case_id"), "variant_id": meta.get("variant_id"), "arm": meta.get("arm"),
                                 "model": meta.get("model"), "status": packet.get("status"), "lawyer": expected,
                                 **{f"judge_{f}": v for f, v in per_judge.items()}, "rubric_version": packet.get("rubric_version")}})
    return out


def step_judgeplayground(api: Api, live: bool, manifest: dict) -> None:
    from dealpoint.eval.judge_run import _system_prompt

    rows = judge_packet_rows()
    print(f"judgeplayground: dataset {JUDGE_PACKETS_DATASET} ({len(rows)} lawyer-scored packets, lawyer scores as expected) + chat prompt "
          f"{JUDGE_PANEL_PROMPT_SLUG} (the frozen M5 rubric); run it in the Playground over {', '.join(JUDGE_PANEL_MODELS.values())}")
    manifest["judgeplayground"] = {"dataset": JUDGE_PACKETS_DATASET, "rows": len(rows), "prompt": JUDGE_PANEL_PROMPT_SLUG, "models": JUDGE_PANEL_MODELS}
    if not live:
        return
    import braintrust
    ds = braintrust.init_dataset(project=PROJECT, name=JUDGE_PACKETS_DATASET,
                                 description="The 24 blinded packets the lawyer scored, rendered exactly as the M5 judges saw them; expected = the lawyer's reasoning/evidence/trajectory/professional (1-5); each judge family's scores in metadata.")
    for r in rows:
        ds.insert(input=r["input"], expected=r["expected"], metadata=r["metadata"], id=r["id"])
    ds.flush()
    project = braintrust.projects.create(name=PROJECT)
    project.prompts.create(name="Judge panel: the M5 rubric", slug=JUDGE_PANEL_PROMPT_SLUG, model=JUDGE_PANEL_MODELS["Mistral"], if_exists="replace",
                           messages=[{"role": "system", "content": _system_prompt()}, {"role": "user", "content": "{{input}}"}],
                           metadata={"purpose": "Playground: the same packet judged by Mistral, NVIDIA and ByteDance side by side, against the lawyer in expected",
                                     "models": JUDGE_PANEL_MODELS})
    project.publish()
    _ledger("playground", "judge-packets+prompt")


def step_playground(api: Api, live: bool, manifest: dict) -> None:
    rows = playground_rows(); prompts = playground_prompts()
    print(f"playground: dataset {PLAYGROUND_DATASET} ({len(rows)} arm-A packets) + {len(prompts)} chat prompts "
          f"({', '.join(p['slug'] for p in prompts)}); scorers judge-{', judge-'.join(PLAYGROUND_JUDGES)}")
    manifest["playground"] = {"dataset": PLAYGROUND_DATASET, "rows": len(rows), "prompts": [p["slug"] for p in prompts]}
    if not live:
        return
    import braintrust
    ds = braintrust.init_dataset(project=PROJECT, name=PLAYGROUND_DATASET,
                                 description="18 judged cases as the exact arm-A packet (question block + the 5 dense passages arm A retrieved). Playground input for the prompt A/B/C.")
    for r in rows:
        ds.insert(input=r["input"], expected=r["expected"], metadata=r["metadata"], id=r["id"])
    ds.flush()
    project = braintrust.projects.create(name=PROJECT)
    for pr in prompts:
        project.prompts.create(name=pr["name"], slug=pr["slug"], messages=pr["messages"], model=WORKHORSE_MODEL, if_exists="replace",
                               metadata={"purpose": "playground arm-A prompt A/B/C", "user_turn": "the arm-A packet from " + PLAYGROUND_DATASET})
    project.publish()
    _ledger("playground", "dataset+prompts")


def _eval_progress(api: Api, experiment_id: str) -> tuple[int, int]:
    """(root rows with an output, scorer spans that produced a score) for an experiment. Scores live on
    the scorer spans (`span_attributes.type == "score"`), not on the root rows the fetch API returns."""
    events = api.fetch_rows(experiment_id)
    roots = [r for r in events if _is_root(r)]
    scored = sum(1 for e in events if (e.get("span_attributes") or {}).get("type") == "score"
                 and any(v is not None for v in (e.get("scores") or {}).values()))
    return sum(1 for r in roots if r.get("output") is not None), scored


def run_playground(api: Api, manifest: dict, judges: tuple[str, ...] = PLAYGROUND_JUDGES, variants: set[str] | None = None,
                   n_rows: int = 18, wait_s: int = 1200) -> None:
    """Pre-run the Playground server-side (POST /v1/eval): one experiment per prompt over the arm-A
    packets, scored by the LLM judges. Scores: len(prompts) x 18 x len(judges); OpenRouter: cents.

    The eval runs asynchronously on Braintrust's side and a non-streaming POST hits the gateway's
    timeout after ~3 minutes while the eval keeps going (observed live), so the request streams, and
    completion is decided by polling the experiment until every row has an output (or `wait_s`).
    An experiment that already exists under the same name would be appended to, not replaced: pass
    `variants` to re-run one, after deleting it."""
    prompts = [pr for pr in playground_prompts() if not variants or pr["slug"].removeprefix("arm-a-prompt-") in variants]
    ds = api.get("dataset", {"project_id": PROJECT_ID, "dataset_name": PLAYGROUND_DATASET}).get("objects", [])
    if not ds:
        raise SystemExit("playground dataset missing; run --only playground first")
    fid = {o["slug"]: o["id"] for o in api.get("function", {"project_id": PROJECT_ID, "limit": 200}).get("objects", [])}
    scores = [{"function_id": fid[f"judge-{d}"]} for d in judges]
    print(f"playground run: {len(prompts)} experiments x {n_rows} rows x {len(judges)} judges = {len(prompts) * n_rows * len(judges)} scores")
    manifest["playground_run"] = []
    for pr in prompts:
        variant = pr["slug"].removeprefix("arm-a-prompt-")
        name = f"playground-arm-A-{variant}"
        body = {"project_id": PROJECT_ID, "data": {"dataset_id": ds[0]["id"]}, "task": {"function_id": fid[pr["slug"]]}, "scores": scores,
                "experiment_name": name, "metadata": {k: v for k, v in classify_experiment(name).items() if k not in ("tags", "description")},
                "max_concurrency": 2, "stream": True}
        with contextlib.suppress(requests.RequestException), \
                requests.post(f"{API}/eval", headers=api.h, json=body, timeout=(30, 600), stream=True) as r:
            for _ in r.iter_lines():
                pass
        exp = api.get("experiment", {"project_id": PROJECT_ID, "experiment_name": name}).get("objects", [])
        if not exp:
            print(f"  {name}: FAILED (no experiment created)")
            manifest["playground_run"].append({"experiment": name, "ok": False})
            continue
        deadline = time.time() + wait_s
        outputs = scored = 0
        while time.time() < deadline:
            outputs, scored = _eval_progress(api, exp[0]["id"])
            if outputs >= n_rows:
                break
            time.sleep(30)
        ok = outputs >= n_rows
        print(f"  {name}: {'complete' if ok else 'INCOMPLETE'} ({outputs}/{n_rows} outputs, {scored} scores written)")
        manifest["playground_run"].append({"experiment": name, "id": exp[0]["id"], "ok": ok, "outputs": outputs, "scores_written": scored})
        _ledger(name, "playground-eval", scored)


JUDGE_DIMENSIONS = ("reasoning", "evidence", "trajectory", "professional")


def judge_scorer_messages(dimension: str) -> list[dict]:
    from dealpoint.eval.rubric import rubric_text
    rubric = rubric_text()
    section = rubric.split(f"## Dimension: {dimension}", 1)
    anchors = section[1].split("## Dimension:", 1)[0].strip() if len(section) > 1 else rubric
    system = ("You are scoring one blinded trace of a merger-agreement question-answering system on ONE dimension: "
              f"{dimension}. Use these anchors exactly.\n\n{anchors}\n\nReply with the single digit 1, 2, 3, 4 or 5.")
    user = ("Question:\n{{input}}\n\nSystem output (finding, trajectory summary):\n{{output}}\n\n"
            "Expert gold span (use only for the evidence dimension):\n{{expected}}")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def step_judges(api: Api, live: bool, manifest: dict) -> None:
    print("judges: 4 LLM scorers via `bt scorers create`: " + ", ".join(f"judge-{d} ({JUDGE_MODELS[d]})" for d in JUDGE_DIMENSIONS))
    manifest["llm_scorers"] = [f"judge-{d}" for d in JUDGE_DIMENSIONS]
    if not live:
        return
    scratch = Path("data/reports/showroom_scorers"); scratch.mkdir(parents=True, exist_ok=True)
    for d in JUDGE_DIMENSIONS:
        msgs = scratch / f"judge-{d}.messages.json"
        msgs.write_text(json.dumps(judge_scorer_messages(d), indent=2), encoding="utf-8")
        cmd = ["bt", "scorers", "create", f"Judge: {d}", "--slug", f"judge-{d}", "--env-file", ENV_FILE, "--no-input", "-p", PROJECT,
               "--description", f"Calibrated rubric, {d} dimension (rubric frozen in Git). Three cheap judges of this shape were calibrated against a lawyer in M5.",
               "--model", JUDGE_MODELS[d], "--messages", f"@{msgs}", "--choice-scores", '{"1":0,"2":0.25,"3":0.5,"4":0.75,"5":1}',
               "--use-cot=true", "--if-exists", "replace"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
        ok = r.returncode == 0
        print(f"  judge-{d}: {'created' if ok else 'FAILED'} {'' if ok else (r.stderr or r.stdout)[-300:]}")
        if ok:
            _ledger("scorers", f"judge-{d}")


def step_views(api: Api, live: bool, manifest: dict) -> None:
    from dealpoint.eval.braintrust_cockpit import _upsert_view, _view_data_for
    views = [
        {"name": "Arms A to D (same 32 cases)", "view_type": "experiments", "btql": "name like '_-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc' or name like '_-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc'",
         "caption": "Four architectures, one model each, same cases. Select all four and open the Summary table."},
        {"name": "Models (arm D fixed)", "view_type": "experiments", "btql": "name like 'D-%-3fcdae7' or name like 'pareto-%' or name = 'D-z-ai_glm-5.3-flash-e2b4a2b97561-e3ee9cc' or name = 'D-anthropic_claude-haiku-4.5-e2b4a2b97561-e3ee9cc'",
         "caption": "Arm D on five models. Sort by $/case to find the frontier."},
        {"name": "Judges and the lawyer", "view_type": "experiments", "btql": "name like 'judge-%' or name like 'judged-%' or name = 'deepeval-crosscheck'",
         "caption": "judge/* aggregates, human/* on the same rows, DeepEval as the second opinion."},
    ]
    print("views: " + ", ".join(v["name"] for v in views))
    manifest["views"] = [v["name"] for v in views]
    if not live:
        return
    from dealpoint.eval.braintrust_cockpit import RestClient
    rc = RestClient(load_braintrust_key())  # type: ignore[arg-type]
    for v in views:
        _upsert_view(rc, "project", PROJECT_ID, v["view_type"], v["name"], _view_data_for({"btql": v["btql"]}))
    _ledger("views", "showroom-views")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    live = "--live" in argv
    only = None
    replay = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))
        if a == "--replay" and i + 1 < len(argv):
            replay = argv[i + 1]
    run_pg = "--run-playground" in argv
    variants = None
    for i, a in enumerate(argv):
        if a == "--variants" and i + 1 < len(argv):
            variants = set(argv[i + 1].split(","))
    key = load_braintrust_key()
    if live and not key:
        print("no BRAINTRUST_API_KEY; refusing a live run", file=sys.stderr)
        return 2
    api = Api(key or "")
    if key:
        os.environ["BRAINTRUST_API_KEY"] = key     # the SDK steps and the org lookup read the env
        api.resolve_project()
    org = org_slug(key)
    manifest = {"mode": "live" if live else "dry-run", "started_at": datetime.now(UTC).isoformat(), "project": PROJECT,
                "project_id": PROJECT_ID, "org": org, "env_file": ENV_FILE, "ledger": str(_ledger_path())}
    print(f"{'LIVE' if live else 'DRY RUN'}: showroom on {PROJECT} in org {org} (scores written: 0 by construction; ledger {_ledger_path()})")
    if replay:
        step_replay(api, live, manifest, replay)
        return 0
    if run_pg:
        if not live:
            print("playground run needs --live (it writes scores and spends OpenRouter cents)")
            return 2
        run_playground(api, manifest, variants=variants)
        return 0
    for name in STEPS:
        if only and name not in only:
            continue
        globals()[f"step_{name}"](api, live, manifest)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    if live:
        out = _manifest_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"manifest -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
