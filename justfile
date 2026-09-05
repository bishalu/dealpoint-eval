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

# regenerate data/reports/four_arm.json + four_arm.md + README Results section
report:
    uv run python -m dealpoint.eval.report

# milestone 4 acceptance gates only, offline
gate-m4:
    uv run pytest -m "gate_m4 and not needs_network and not needs_model" -q

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
