# SSSF starter recipes. Stamped by install.py, then yours to edit.
#
# Deliberately small. These are the handful you need on day one: run something,
# watch it, and open the trace. Add your own as your chains grow, and see the
# example branch for the fuller set (orchestrator agents, kill, rosters, ipi).

# `.env` reaches every ADW through this, so keys work without exporting them.
set dotenv-load
set positional-arguments

# Every recipe passes this through, so `SSSF_CONFIG=other.yaml just sdlc "..."`
# swaps the whole roster for one run.
config := env_var_or_default("SSSF_CONFIG", "adws/adw_sssf_config/sssf.config.yaml")
db     := "adws/adw_data/sssf.db"

# list every recipe
default:
    @just --list

# ── first run ───────────────────────────────────────────────────────────────

# Proves the whole path works: config validated, session minted, agent ran,
# envelope parsed, gates checked, trace written. Costs a few cents and changes
# nothing in your repo, because both workflows are read-only.
#
# (`just --list` shows only the LAST comment line, so that one is the summary.)

# start here: two cheap read-only runs, end to end
demo:
    @echo "1/2  adw_prompt: one agent, one prompt"
    uv run adws/adw_prompt.py --config {{config}} --agent scout "reply with a one-line summary of this repo"
    @echo "\n2/2  adw_scout: read-only recon"
    uv run adws/adw_scout.py --config {{config}} "list the top-level directories in this repo and what each is for. change nothing."
    @echo "\nboth done. now run:  just sessions    (or: just obs)"

# ── run a workflow ──────────────────────────────────────────────────────────
# Args pass straight through: "<prompt or path/to/prompt.md>" [--adw-id X]

# one agent, one prompt: just prompt "summarize this repo"
prompt *ARGS:
    uv run adws/adw_prompt.py --config {{config}} "$@"

# read-only recon: just scout "where is auth handled"
scout *ARGS:
    uv run adws/adw_scout.py --config {{config}} "$@"

# plan only: just plan "add a /health endpoint"
plan *ARGS:
    uv run adws/adw_plan.py --config {{config}} "$@"

# planner, builder, commit: just plan-build "add a /health endpoint"
plan-build *ARGS:
    uv run adws/adw_plan_build.py --config {{config}} "$@"

# plan, build, test, commit: just sdlc "add a /health endpoint"
sdlc *ARGS:
    uv run adws/adw_plan_build_test.py --config {{config}} "$@"

# the full chain, plus review and docs: just simple-sdlc "add a /health endpoint"
simple-sdlc *ARGS:
    uv run adws/adw_simple_sdlc.py --config {{config}} "$@"

# ── headroom (MCP-only context tools, feature-flagged) ──────────────────────
# `SSSF_HEADROOM=1 just sdlc "..."` turns the flag on for one launch; the roster
# default is `defaults.headroom.enabled` in sssf.config.yaml. docs/headroom.md.

# prove a Claude worker can see and call Headroom, and measure baseline vs flag on
headroom-probe *ARGS:
    python3 adws/headroom_integration_test.py "$@"

# ── watch it ────────────────────────────────────────────────────────────────
# Reads never block a running workflow, the db is WAL. Poll as hard as you like.

# the last 10 runs
sessions:
    @sqlite3 {{db}} "select adw_id, status, substr(request,1,50), total_tokens, round(total_cost,4) from sessions order by started_at desc limit 10;"

# phase status in sequence: just phases <adw_id>
phases ADW_ID:
    @sqlite3 {{db}} "select seq, name, kind, owner, status, attempt from phases where adw_id='{{ADW_ID}}' order by seq;"

# the live event tail: just tail <adw_id>
tail ADW_ID:
    @sqlite3 {{db}} "select rowid, type, name, started_at from events where adw_id='{{ADW_ID}}' order by rowid desc limit 25;"

# what a run has alive right now, with pids: just procs <adw_id>
procs ADW_ID:
    @sqlite3 {{db}} "select kind, name, pid, command, started_at from processes where adw_id='{{ADW_ID}}' and ended_at is null order by id;"

# ── observability UI ────────────────────────────────────────────────────────

# Needs bun. The db path is passed explicitly because the server runs from the
# app dir and would otherwise look for a trace db sitting next to itself.

# boot the trace UI, http://localhost:4601 (api on :4600)
obs:
    cd .claude/skills/sssf/apps/visualizer && bun install && (SSSF_DB={{justfile_directory()}}/{{db}} bun run server/index.ts &) && bunx vite

# ── dealpoint (M0 data foundation) ─────────────────────────────────────────

# rebuild the MAUD data foundation: canonical texts, sections, alignment, cases
data:
    uv run python -m dealpoint.cli data

# full offline test suite (no network, no model spend)
test:
    uv run pytest -m "not needs_network and not needs_model" -q

# milestone 0 acceptance gates only, offline
gate-m0:
    uv run pytest -m "gate_m0 and not needs_network and not needs_model" -q

# build the dense retrieval index (20 selected + 30 redacted documents)
index:
    uv run python -m dealpoint.corpus.build_index

# milestone 1 acceptance gates only, offline (excludes the metered smoke)
gate-m1:
    uv run pytest -m "gate_m1 and not needs_network and not needs_model" -q

# run the agent CLI: just agent --case contract_0__q01 --arm B --model anthropic/claude-haiku-4.5
agent *ARGS:
    uv run python -m dealpoint.agent.run "$@"

# score one arm x model over a case set: just eval B anthropic/claude-haiku-4.5 dev --limit 5
eval ARM MODEL SET *ARGS:
    uv run python -m dealpoint.eval.run --arm {{ARM}} --model {{MODEL}} --set {{SET}} "$@"

# 3 dev cases, Haiku, arm B, offline scoring; --send publishes those same rows to Braintrust
smoke *ARGS:
    uv run python -m dealpoint.eval.smoke "$@"

# print the JSON cost estimate for a sweep: just budget four_arm
budget SWEEP:
    uv run python -m dealpoint.eval.budget {{SWEEP}}

# milestone 2 acceptance gates only, offline (excludes the metered smoke)
gate-m2:
    uv run pytest -m "gate_m2 and not needs_network and not needs_model" -q

# run the LLM-free retrieval tournament over the dev set: just tournament [--limit 10]
tournament *ARGS:
    uv run python -m dealpoint.eval.tournament "$@"

# milestone 3 acceptance gates only, offline
gate-m3:
    uv run pytest -m "gate_m3 and not needs_network and not needs_model" -q

# ── dealpoint (M4 four-arm experiment) ──────────────────────────────────────

# regenerate the frozen discriminative subset: data/eval/test_subset_v1.json
subset:
    uv run python -m dealpoint.eval.subset

# run the M4 metered sweeps in spec order (dev smoke, caching probe, GLM headline, Haiku replication)
sweep-m4 *ARGS:
    uv run python -m dealpoint.eval.four_arm_sweep "$@"

# M4.1: 2-case diagnostic probes (arm D@Haiku, arm A/B@GLM) -- before (free) + after (metered)
probe-m4-1 *ARGS:
    uv run python -m dealpoint.eval.probe_m4_1 "$@"

# M4.1: v2 four-arm re-run after the harness repair (preserves v1 artefacts first)
sweep-m4-1 *ARGS:
    uv run python -m dealpoint.eval.four_arm_sweep --v2 "$@"

# regenerate data/reports/four_arm.json + four_arm.md + README Results section
report:
    uv run python -m dealpoint.eval.report

# milestone 4 acceptance gates only, offline
gate-m4:
    uv run pytest -m "gate_m4 and not needs_network and not needs_model" -q

# ── dealpoint (M5 calibrated multi-judge evaluation) ────────────────────────

# M5: verify the judge slate, then judge the frozen judged subset (METERED)
judge *ARGS:
    uv run python -m dealpoint.eval.judge_run "$@"

# M5: rebuild the calibration package and data/reports/judges.json (offline, no model calls)
calibration:
    uv run python -m dealpoint.eval.calibration

# milestone 5 acceptance gates only, offline
gate-m5:
    uv run pytest -m "gate_m5 and not needs_network and not needs_model" -q

# ── dealpoint (M6 model cost/quality Pareto experiment) ─────────────────────

# M6: verify the Pareto slate at run time (2 dev cases per candidate, METERED)
pareto-slate *ARGS:
    uv run python -m dealpoint.eval.pareto_slate "$@"

# M6: verify the Pareto slate (probes), then sweep arm D per model (METERED)
pareto-sweep *ARGS:
    uv run python -m dealpoint.eval.pareto_sweep "$@"

# M6: regenerate data/reports/pareto.json + pareto.md + pareto.svg + README block
pareto-report:
    uv run python -m dealpoint.eval.pareto_report

# milestone 6 acceptance gates only, offline
gate-m6:
    uv run pytest -m "gate_m6 and not needs_network and not needs_model" -q

# ── dealpoint (M7a framework roles) ─────────────────────────────────────────

# M7a: re-measure and record the optional dependency-group disk guard
disk-guard *ARGS:
    uv run python -m dealpoint.eval.disk_guard "$@"

# M7a: LlamaIndex RAG lab -- native eval on canonical dev retrieval + synthetic-query study
li-rag-eval *ARGS:
    uv run python -m dealpoint.rag_lab.report "$@"

# M7a: generate the frozen synthetic dev query set (METERED, cheap workhorse)
synth-queries *ARGS:
    uv run python -m dealpoint.rag_lab.synthetic "$@"

# M7a: DeepEval independent cross-check on the judged subset (METERED)
deepeval-crosscheck *ARGS:
    uv run python -m dealpoint.eval.deepeval_adapter "$@"

# M7a: recreate every Braintrust artifact from Git/local sources; idempotent.
# Dry run by default; pass --live to write (scores cost money on this plan).
braintrust-sync *ARGS:
    uv run python -m dealpoint.eval.braintrust_sync "$@"

# M7a: offline, zero-network regeneration of braintrust_sync.json from current code
braintrust-sync-dry-run:
    uv run python -m dealpoint.eval.braintrust_sync --dry-run

# M7a: run the six BTQL investigations and save results
btql *ARGS:
    uv run python -m dealpoint.eval.btql "$@"

# M7a: regenerate the standalone framework_versions.json artifact
framework-versions *ARGS:
    uv run python -m dealpoint.eval.framework_versions "$@"

# milestone 7 acceptance gates only, offline
gate-m7:
    uv run pytest -m "gate_m7 and not needs_network and not needs_model" -q

# ── dealpoint (M7b Braintrust cockpit + multi-judge demo) ──────────────────

# M7b: create/update the cockpit (views, dashboard, Topics, one Pattern, hero-case replay, human/ scores); idempotent.
# Dry run by default; `just braintrust-cockpit --live` writes, capped at LIVE_SCORE_CAP scores and never re-logging.
braintrust-cockpit *ARGS:
    uv run python -m dealpoint.eval.braintrust_cockpit "$@"

# M7b: offline, zero-network regeneration of demo_manifest.json from current code
braintrust-cockpit-dry-run:
    uv run python -m dealpoint.eval.braintrust_cockpit --dry-run

# Demo showroom: tags/metadata, row metadata, Logs, review flags, parameters, prompts, LLM judge scorers, views,
# Playground dataset+prompts; org resolved from the key, ledger under data/reports/orgs/<org>/. Dry run by default.
#   just braintrust-showroom --live                                   # everything (0 scores)
#   just braintrust-showroom --live --only logs                       # one step
#   just braintrust-showroom --live --replay contract_144__q05:D@glm  # one new log, for online scoring
#   just braintrust-showroom --live --run-playground                  # pre-run the prompt A/B/C (scores + cents)
braintrust-showroom *ARGS:
    uv run python -m dealpoint.eval.braintrust_showroom "$@"

# M7b: regenerate docs/demo-walkthrough.md from data/reports/demo_manifest.json
demo-walkthrough:
    uv run python -m dealpoint.eval.demo_walkthrough

# M7b: record one cockpit-session step's result: just demo-manifest-record --step 2 --note "..." --url https://...
demo-manifest-record *ARGS:
    uv run python -m dealpoint.eval.demo_manifest_record "$@"

# milestone 7b acceptance gates only, offline
gate-m7b:
    uv run pytest -m "gate_m7b and not needs_network and not needs_model" -q

# ── dealpoint (M9 MLflow mirror of the Braintrust showroom) ────────────────

# M9: self-hosted MLflow tracking server, SQLite backend, artifacts under data/mlflow/ (gitignored)
# MLFLOW_PUBLIC_HOST is the exe.dev proxy name (balpad.exe.xyz); MLflow rejects any other Host header
# (its DNS-rebinding guard), so the proxy's name must be allowed explicitly.
mlflow-server:
    uv run --extra mlflow mlflow server --backend-store-uri sqlite:///data/mlflow/mlflow.db \
      --artifacts-destination data/mlflow/artifacts --host 127.0.0.1 --port 5000 \
      --allowed-hosts "${MLFLOW_PUBLIC_HOST:-balpad.exe.xyz},${MLFLOW_PUBLIC_HOST:-balpad.exe.xyz}:5000,127.0.0.1:5000,localhost:5000" \
      --cors-allowed-origins "https://${MLFLOW_PUBLIC_HOST:-balpad.exe.xyz},https://${MLFLOW_PUBLIC_HOST:-balpad.exe.xyz}:5000"

# M9: recreate every MLflow artifact from Git/local sources; idempotent. Dry run by default; --live writes.
mlflow-sync *ARGS:
    uv run python -m dealpoint.eval.mlflow_mirror "$@"

# M9: offline, zero-network regeneration of the mlflow mirror plan from current code
mlflow-sync-dry-run:
    uv run python -m dealpoint.eval.mlflow_mirror --dry-run

# M9b: the config decision run tree (nested runs, Pareto artifacts, registry versions). No model calls.
mlflow-decision *ARGS:
    uv run python -m dealpoint.eval.mlflow_decision "$@"

# M9 D3: judge alignment against the lawyer's 24 packets (MLflow-unique, spend-gated, cap $1.50)
mlflow-align-judges *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges align "$@"

# M9 D4: GEPA prompt optimization on arm-a-prompt-base (MLflow-unique, spend-gated, cap $1.00)
mlflow-optimize-prompt *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges optimize "$@"

# M9: batch stand-in for Databricks-only online scoring -- re-score traces newer than --since
mlflow-score-new *ARGS:
    uv run python -m dealpoint.eval.mlflow_judges score-new "$@"

# milestone 9 acceptance gates only, offline
gate-m9:
    uv run pytest -m "gate_m9 and not needs_network and not needs_model" -q

# ── M9c: every MLflow tab filled, and the eight dashboards ported word for word ────────────────

# M9c D11-D15: sessions, judges, review queues, agent versions, gateway+playground. Dry run by default.
mlflow-tabs *ARGS:
    uv run python -m dealpoint.eval.mlflow_tabs "$@"

# M9c D12: re-log one stored trace as a new trace tagged category=live-replay, e.g. `just mlflow-replay contract_144__q05:D@glm --live`
mlflow-replay CASE_VARIANT *ARGS:
    uv run python -m dealpoint.eval.mlflow_tabs replay {{CASE_VARIANT}} "$@"

# M9c D16: the eight dashboards' numbers, offline (local evaluator) by default; --live runs real, read-only BTQL
braintrust-dashboard-snapshot *ARGS:
    uv run python -m dealpoint.eval.mlflow_dashboards snapshot "$@"

# M9c D16: log the eight dashboard runs (HTML artifact + metrics) to MLflow. Dry run by default.
mlflow-dashboards *ARGS:
    uv run python -m dealpoint.eval.mlflow_dashboards runs "$@"

# milestone 9c acceptance gates only, offline
gate-m9c:
    uv run pytest -m "gate_m9c and not needs_network and not needs_model" -q

# milestone 9d acceptance gates only, offline
gate-m9d:
    uv run pytest -m "gate_m9d and not needs_network and not needs_model" -q

# ── mvp orchestration (project-level autonomous loop) ──────────────────────

# where the autonomous run stands; launches nothing
mvp-status:
    uv run adws/adw_mvp.py --dry-run

# the core eval MVP loop, M0.1 → M6, resumable from specs/mvp/state.json: just mvp [--only m1]
mvp *ARGS:
    uv run adws/adw_mvp.py --config {{config}} "$@"

# one milestone in its own session: just milestone --milestone m1
milestone *ARGS:
    uv run adws/adw_milestone.py --config {{config}} "$@"

# stop a run cleanly — coding agents first, then the workflow — from the pids the trace recorded: just kill <adw_id>
kill ADW_ID:
    @sqlite3 {{db}} "select pid from processes where adw_id='{{ADW_ID}}' and ended_at is null order by case kind when 'agent' then 0 else 1 end, id desc;" | while read pid; do if kill -0 "$pid" 2>/dev/null; then echo "TERM $pid"; kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid"; fi; done; sleep 5; \
    sqlite3 {{db}} "select pid from processes where adw_id='{{ADW_ID}}' and ended_at is null;" | while read pid; do if kill -0 "$pid" 2>/dev/null; then echo "KILL $pid"; kill -KILL "$pid"; fi; done; \
    sqlite3 {{db}} "update processes set ended_at=datetime('now') where adw_id='{{ADW_ID}}' and ended_at is null; update sessions set status='fail', ended_at=coalesce(ended_at, datetime('now')) where adw_id='{{ADW_ID}}' and status='running';"; \
    echo "stopped {{ADW_ID}}"
