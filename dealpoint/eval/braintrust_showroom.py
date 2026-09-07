"""Braintrust showroom: make the existing `dealpoint-eval` project walkable tab by tab, for free.

Projects are free and scores cost money, and the scores this project already holds were paid for.
So this module never writes a score. It only adds the things Braintrust does not charge scores for:

    tag      experiment-level metadata, tags and descriptions, so the Experiments tab reads as a story
    rows     row-level metadata merged onto existing rows (arm, model, variant, case_type), so
             group-by and filters work; merge-updates on `metadata` do not touch scores
    logs     the 108 judged traces plus the six representative ones replayed into Logs as full
             span trees with NO scores (Topics, Patterns, Debugger and Loop need Logs, not experiments)
    review   the 12-trace review set flagged for human review on the judge experiments
    params   four Parameters objects, one per arm, so "change one parameter, rerun, compare" is a real object
    prompts  three arm-A prompt variants for the Playground A/B/C, next to the base prompt
    judges   the calibrated rubric as four Braintrust LLM scorers on the OpenRouter judge models (your key,
             your ledger; creation is free, invocation is an OpenRouter call) via `bt scorers create`
    views    saved comparison views: arms, models, judges, RAG

Dry run is the default and prints the plan; `--live` executes. Every live write is recorded in the
score ledger with n_scores=0, and a guard asserts no payload ever carries a `scores` key.

    uv run python -m dealpoint.eval.braintrust_showroom              # dry run
    uv run python -m dealpoint.eval.braintrust_showroom --live       # execute
    uv run python -m dealpoint.eval.braintrust_showroom --live --only tag,rows
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
STEPS = ("tag", "rows", "logs", "review", "params", "prompts", "judges", "views")

_ARM = re.compile(r"^(?P<arm>[ABCD])-(?P<model>.+)-e2b4a2b97561-(?P<sha>[0-9a-f]{7})$")
_JUDGED = re.compile(r"^judged-(?P<arm>[ABCD])-(?P<model>.+)-e2b4a2b97561-(?P<sha>[0-9a-f]{7})$")
_JUDGE = re.compile(r"^judge-(?P<variant>[ABCD]@.+)$")
_PARETO = re.compile(r"^pareto-(?P<model>.+)$")
_RAG = re.compile(r"^rag-(?P<config>.+)$")
GLM = "z-ai/glm-5.3-flash"


def _model_from_slug(slug: str) -> str:
    return slug.replace("_", "/", 1)


def classify_experiment(name: str) -> dict:
    """Experiment name -> {stage, family, arm, model, tags, description}. Pure, testable."""
    if m := _ARM.match(name):
        arm, model, sha = m["arm"], _model_from_slug(m["model"]), m["sha"]
        stage = "arms" if model in (GLM, "anthropic/claude-haiku-4.5") else "models"
        run = {"e3ee9cc": "M4/M6 canonical run (32-case test subset)", "4d2e361": "M4 first run (original)",
               "3fcdae7": "M6 Pareto sweep", "c96758b": "M2 smoke", "a357949": "M2 smoke"}.get(sha, sha)
        return {"stage": stage, "family": "agent", "arm": arm, "model": model, "git_sha": sha,
                "tags": [f"stage:{stage}", f"arm:{arm}", f"model:{model}"],
                "description": f"Arm {arm} on {model}. {run}. Same 32 test cases as its siblings; compare case by case."}
    if m := _JUDGED.match(name):
        arm, model = m["arm"], _model_from_slug(m["model"])
        return {"stage": "evaluation", "family": "judged-original", "arm": arm, "model": model, "git_sha": m["sha"],
                "tags": ["stage:evaluation", f"arm:{arm}", f"model:{model}", "milestone:M5"],
                "description": f"M5 original judged run, arm {arm} on {model}: 18 judged cases, mean-of-three-judges scores."}
    if m := _JUDGE.match(name):
        variant = m["variant"]; arm, short = variant.split("@", 1)
        return {"stage": "evaluation", "family": "judge", "arm": arm, "model": short, "variant_id": variant,
                "tags": ["stage:evaluation", f"arm:{arm}", f"variant:{variant}", "milestone:M7"],
                "description": f"Judge scores (judge/<dimension>, three families averaged) and the lawyer's human/<dimension> scores for {variant} on the 18 judged cases."}
    if m := _PARETO.match(name):
        model = _model_from_slug(m["model"])
        return {"stage": "economics", "family": "pareto", "arm": "D", "model": model,
                "tags": ["stage:economics", "arm:D", f"model:{model}", "milestone:M6"],
                "description": f"M6 Pareto sweep: arm D fixed, model {model}. obj/* scores plus $/case in metadata."}
    if m := _RAG.match(name):
        return {"stage": "rag", "family": "rag", "config": m["config"],
                "tags": ["stage:rag", f"config:{m['config']}", "milestone:M3/M7a"],
                "description": f"RAG lab: retriever config {m['config']} on the 58 dev queries; li/* (LlamaIndex) next to obj/* (MAUD gold spans)."}
    if name == "deepeval-crosscheck":
        return {"stage": "evaluation", "family": "deepeval", "tags": ["stage:evaluation", "milestone:M7a"],
                "description": "DeepEval's independent read on the 108 judged traces (task completion, tool correctness, step efficiency)."}
    if name == "m7b-hero-case":
        return {"stage": "traces", "family": "hero", "tags": ["stage:traces", "milestone:M7b"],
                "description": "The hero case: three judge spans plus aggregate under scoring, for A@haiku and D@haiku, replayed with zero model calls."}
    if name == "m7-representative-traces":
        return {"stage": "traces", "family": "representative", "tags": ["stage:traces", "milestone:M7b"],
                "description": "Six rule-chosen representative traces, one per failure shape, score-free."}
    return {"stage": "other", "family": "other", "tags": ["stage:other"], "description": ""}


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


def _assert_scoreless(events: list[dict]) -> None:
    for e in events:
        if "scores" in e:
            raise RuntimeError("showroom refuses to write scores; a payload carried a 'scores' key")


def _ledger_path() -> Path:
    return LEDGER_PATH or org_report_path("braintrust_score_ledger.jsonl", override_env="BRAINTRUST_LEDGER_FILE")


def _manifest_path() -> Path:
    return MANIFEST_PATH or org_report_path("showroom_manifest.json")


def _ledger(experiment: str, key: str) -> None:
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"experiment": experiment, "key": key, "n_scores": 0,
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
        from dealpoint.eval.cases import find_case
        from dealpoint.eval.questions import resolve_question
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
        plan.append({"id": e["id"], "name": e["name"], **{k: c[k] for k in ("stage", "family", "tags")}})
        if live:
            body = {"metadata": {**(e.get("metadata") or {}), **{k: v for k, v in c.items() if k not in ("tags", "description")}},
                    "tags": c["tags"], "description": c["description"]}
            api.patch(f"experiment/{e['id']}", body)
    manifest["experiments"] = plan
    by_stage: dict[str, int] = {}
    for p in plan:
        by_stage[p["stage"]] = by_stage.get(p["stage"], 0) + 1
    print(f"tag: {len(plan)} experiments by stage {by_stage}")


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
            is_root = r.get("is_root") if r.get("is_root") is not None else not r.get("span_parents")
            if not is_root:
                continue
            case_id = r.get("input") if isinstance(r.get("input"), str) else (r.get("metadata") or {}).get("case_id")
            meta = {k: v for k, v in c.items() if k in ("stage", "family", "arm", "model", "variant_id", "config")}
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


def step_logs(api: Api, live: bool, manifest: dict) -> None:
    from dealpoint.corpus.document import load_document
    from dealpoint.eval.braintrust_cockpit import _judged_subset, _load_jsonl, _variant_results
    from dealpoint.eval.braintrust_sync import _emit_span_tree, log_hierarchy, representative_cases
    from dealpoint.eval.cases import find_case, resolve_document_id

    subset = _judged_subset()
    plan: list[dict] = []
    for variant in subset.get("variants", []):
        rows = _variant_results(subset, variant["variant_id"])
        for case_id in subset.get("case_ids", list(rows)):
            plan.append({"case_id": case_id, "variant_id": variant["variant_id"], "category": "judged", "row": rows.get(case_id)})
    for sel in representative_cases().get("selections", []):
        rows = {r.get("case_id"): r for r in _load_jsonl(sel["results_path"])} if sel.get("results_path") else {}
        plan.append({"case_id": sel["case_id"], "variant_id": sel["variant_id"], "category": sel["category"], "row": rows.get(sel["case_id"])})
    seen = _ledger_keys()
    todo = [p for p in plan if ("logs", f"{p['case_id']}:{p['variant_id']}:{p['category']}") not in seen]
    print(f"logs: {len(plan)} trace trees planned, {len(todo)} not yet in the ledger, 0 scores")
    manifest["logs"] = {"planned": len(plan), "written_this_run": 0, "roots": []}
    if not live or not todo:
        return
    import braintrust
    logger = braintrust.init_logger(project=PROJECT, set_current=True)
    for p in todo:
        row = p["row"] or {"case_id": p["case_id"], "scores": {}, "record": {"trajectory": []}}
        try:
            case = find_case(p["case_id"]); doc = load_document(resolve_document_id(case))
        except (KeyError, FileNotFoundError):
            case, doc = {}, None
        hierarchy = log_hierarchy(row, case, doc, judge_dims=None)
        arm, _, model = p["variant_id"].partition("@")
        meta = {"case_id": p["case_id"], "variant_id": p["variant_id"], "arm": arm, "model": model, "category": p["category"],
                "status": (row.get("record") or {}).get("status") or row.get("status"),
                "obj_grounded_accuracy": (row.get("scores") or {}).get("grounded_accuracy"), **_case_info(p["case_id"])}
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
        manifest["logs"]["roots"].append({"case_id": p["case_id"], "variant_id": p["variant_id"], "category": p["category"], "root_span_id": getattr(root, "id", None)})
        _ledger("logs", f"{p['case_id']}:{p['variant_id']}:{p['category']}")
        manifest["logs"]["written_this_run"] += 1
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
        rows = {r.get("input"): r["id"] for r in api.fetch_rows(exps[exp_name]) if r.get("span_id") == r.get("root_span_id")}
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
    rc = RestClient(load_braintrust_key())
    for v in views:
        _upsert_view(rc, "project", PROJECT_ID, v["view_type"], v["name"], _view_data_for({"btql": v["btql"]}))
    _ledger("views", "showroom-views")


def main(argv: list[str] | None = None) -> int:
    argv = list(argv) if argv is not None else sys.argv[1:]
    live = "--live" in argv
    only = None
    for i, a in enumerate(argv):
        if a == "--only" and i + 1 < len(argv):
            only = set(argv[i + 1].split(","))
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
