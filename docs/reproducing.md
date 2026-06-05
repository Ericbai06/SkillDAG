# Reproducing the SkillDAG paper

End-to-end walkthrough from a fresh clone. Estimated wall time: ~30 min for
setup + variable per-benchmark runtime (smoke ≈ 10 min, full SkillsBench
scale=1000 ≈ several hours of API + CPU).

## 0. Prerequisites

- Python ≥ 3.10 (python3.11 recommended and used by the scripts when present)
- Git
- Docker (for SkillsBench tasks)
- `gettext` (for `envsubst`) — `brew install gettext` on macOS, `apt install gettext-base` on Debian/Ubuntu
- An OpenAI-compatible embedding API key
- An OpenAI-compatible chat API key for the benchmark agents

## 1. Prepare `.env` and install the SkillDAG library

```bash
git clone https://github.com/Ericbai06/SkillDAG.git
cd SkillDAG
bash scripts/prepare_env.sh          # detects python3.11/PATH and creates .env
# fill API keys in .env
bash scripts/setup.sh                # creates .venv, pip install -e ".[repro,alfworld]", data setup
```

Verify:

```bash
source .env
skilldag help
"${PYTHON}" -m unittest discover -s tests
```

`setup.sh` performs:

1. Creates `.venv/` when needed and installs the package in editable mode with reproduction dependencies:
   `pip install -e ".[repro,alfworld]"`
2. Runs the bundled data downloader → `data/skillsets/skills_{200,500,1000,2000}/`,
   `data/gos_workspace/...`, `data/tasks/tasks/...`
3. Builds `data/alfworld_skills/`, a narrow symlink pool containing only the
   37 ALFWorld skills used by `skillgraph_alfworld.json` when a packaged pool is
   not already present
4. Downloads published SkillDAG graph artifacts into `data/skilldag_graphs/`

If you prefer not to use the published SkillDAG `skillgraph.json` files,
initialize them locally as described in §3a.

For maintenance-only checks, `bash scripts/setup.sh --skip-install --skip-data`
verifies the local interpreter/PATH without downloading data or installing deps.

**Data download priority:** Run `bash scripts/download_data.sh --graphs` first — it
downloads the authoritative paper artifacts (skill graphs + embeddings) from
`Eric068/SkillDAG` on HuggingFace. Skillsets and tasks are downloaded automatically
by `setup.sh` or can be fetched separately with `download_data.sh --skillsets --tasks`. 

## 3. Install the Harbor / SkillsBench framework

The Harbor binary is the test runner used by the SkillsBench arm. It lives in
the upstream [`benchflow-ai/skillsbench`](https://github.com/benchflow-ai/skillsbench)
repository.

```bash
git clone https://github.com/benchflow-ai/skillsbench.git
cd skillsbench
uv sync                              # or: python -m venv .venv && pip install -e .
cd -                                  # back to the SkillDAG repo
bash scripts/prepare_env.sh           # adds Harbor/envsubst/codex paths to .env
```

If your platform requires it, follow the upstream Harbor install notes for
Docker permissions and per-OS setup.

## 3a. Initialize your own SkillDAG graphs instead of downloading the published ones

```bash
# Per scale (one-off, ~$0.02/skill for embeddings + LLM pair classification)
for scale in 200 500 1000 2000; do
  skills_dir="data/skillsets/skills_${scale}"
  [ -d "${skills_dir}/skills_${scale}" ] && skills_dir="${skills_dir}/skills_${scale}"
  skilldag initialize-graph \
    --skills-dir "${skills_dir}" \
    --graph-path data/skilldag_graphs/skillgraph_${scale}.json
done

# For ALFWorld
skilldag initialize-graph \
  --skills-dir data/alfworld_skills \
  --graph-path data/skilldag_graphs/skillgraph_alfworld.json
```

This step requires both `SKILLDAG_EMBEDDING_API_KEY` and `SKILLDAG_LLM_API_KEY`
(or the equivalent provider credentials) in `.env`.

## 4. Run the benchmarks

### SkillsBench

```bash
SKILLDAG_SCALE=200 SKILLDAG_WORKERS=3 bash scripts/run_skillsbench.sh     # smoke (~tens of min)
SKILLDAG_SCALE=1000 SKILLDAG_WORKERS=5 bash scripts/run_skillsbench.sh  # paper scale (~hours)
```

Outputs land in `results/skillsbench/<scale>_w<workers>_<stamp>/`. The script
auto-scores via `analysis/score_skillsbench_gos.py` at the end.

**Paper settings:** scale=1000, workers=5, n_attempts=2 (hardwired in config).

### ALFWorld

`scripts/setup.sh` installs the Python-side ALFWorld dependencies
(`alfworld`, `litellm`, `pyyaml`, `retry`). Point `ALFWORLD_DATA` in `.env` at
the upstream ALFWorld data directory before running this arm. If the default
`data/alfworld/data` directory does not exist yet, run the upstream
`alfworld-download` command first.

```bash
MAX_GAMES=10 bash scripts/run_alfworld.sh                                 # smoke
bash scripts/run_alfworld.sh                                              # full 140-task dev split
```

Outputs land in `results/alfworld/<exp_name>/`.

## 5. Analyze

```bash
python analysis/analyze_skilldag_run.py <results dir>     # graph-edit + retrieval metrics
python analysis/backfill_trial_logs.py  <results dir>     # restitch per-trial logs (post-hoc)
```

### Cross-scale retrieval metrics

Requires all 4 scales to be run first:

```bash
# Build query manifest from any completed run
python scripts/replay_queries.py --mode build-manifest \
  --trials-dir results/skillsbench/<exp>/ \
  --gold-root results/skillsbench_tasks/tasks_skilldag_full_<scale>/ \
  --output analysis/queries.json

# Rerun all queries against each scale's graph
python scripts/replay_queries.py --mode cross-scale \
  --scales 200 500 1000 2000 \
  --graph-dir data/skilldag_graphs \
  --query-manifest analysis/queries.json \
  --output analysis/cross_scale_scores.json
```

### Cold-vs-edited replay

```bash
# Compare cold graph vs train-edited graph on the same queries
python scripts/replay_queries.py --mode edit-effect \
  --cold-graph data/skilldag_graphs/skillgraph_1000.json \
  --edited-graph results/alfworld/traintest_<ts>/graph_after_train.json \
  --query-manifest analysis/queries.json \
  --output analysis/edit_effect_metrics.json
```

## 6. Verify against expected results

Compare your scores against the expected values in `artifacts/expected/`:

```bash
# Compare SkillsBench score
python analysis/score_skillsbench_gos.py results/skillsbench/<exp>/ --json | \
  python3 -c "import json,sys; d=json.load(sys.stdin); print(f'R={d[\"score_R_percent\"]}%')"
```

## Troubleshooting

- **"harbor: command not found"** — Step 3 didn't complete, or `.env` PATH is stale; rerun `bash scripts/prepare_env.sh`.
- **"Initialization requires SKILLDAG_EMBEDDING_API_KEY"** — `.env` not loaded; rerun `bash scripts/prepare_env.sh` and fill the key.
- **"ALFWorld dependencies missing"** — run `bash scripts/setup.sh` or `${PYTHON} -m pip install -e ".[repro,alfworld]"`.
- **"ALFWORLD_DATA directory missing"** — run `alfworld-download` and update `ALFWORLD_DATA` in `.env`.
- **Docker pull/build hangs** — see SkillsBench upstream docs; Harbor caches images, so rebuild is rare on subsequent runs.
- **Skill graph missing for scale X** — re-run §3a for that scale.
- **Graph node has no matching `SKILL.md`** — the run scripts now fail early
  when graph and skill pool do not match. For SkillsBench, point at the inner
  `data/skillsets/skills_<N>/skills_<N>` directory if your skill archive is nested.
  For ALFWorld, use `data/alfworld_skills` or set `SKILLS_DIR` to a 37-skill
  ALFWorld-only directory.
