# Project layout

Every file in the local project folder (`~/Documents/aamas-paper`), what it is, and whether it's in git.
As of 2026-11-08. For what the code does, see [CODEBASE.md](CODEBASE.md).

Legend: **git** = committed and pushed to GitHub · **local** = only on this laptop (and the cluster) · **ignored** = blocked by `.gitignore`

---

## How it's organized

```
aamas-paper/
├── README.md                   overview for new contributors (start here)
├── CLAUDE.md                   rules for Claude (must stay at the top level)
├── pyproject.toml, uv.lock     environment (uv sync); requirements.txt for pip
├── docs/                       all project writing
│   ├── SPEC.md                 project plan, status, results (source of truth)
│   ├── CODEBASE.md             code guide with line links
│   ├── FILES.md                this file
│   ├── reading-list.md         related papers
│   └── initial-ideas.md        original abstract brainstorm (not in git)
├── src/ccsteer/                the library: model I/O, directions, judge, two-agent harness
├── scripts/                    one script per experiment, uses src/ccsteer
├── slurm/                      cluster job files that run the scripts
├── data/                       hand labels for checking the judge
└── results/<experiment>_<model>/   outputs, one folder per run
```

The flow is: **`slurm/` job → runs a `scripts/` file → which uses `src/ccsteer/` → and writes to `results/`**. Results get copied back from the cluster with rsync (SPEC §9).

Where things live:
- **This laptop**: everything below.
- **GitHub** ([airjan-airlines/subagent-misallignment](https://github.com/airjan-airlines/subagent-misallignment), public): everything marked git.
- **Cluster** (`ocf-hpc:~/ccsteer`): same code and results, plus `.venv/` (made by `uv sync`), `logs/` and the models in `~/hf_cache`. Not a git repo.

---

## Top level

| File | Status | What it is |
|---|---|---|
| `README.md` | git | Overview for new contributors: central question, approaches, what is built, initial results, links to everything else |
| `CLAUDE.md` | git | Rules for Claude working in this repo: keep SPEC current, use Slurm, keep detection data separate from the data used to make the model misbehave. Stays at the top level because Claude Code only reads it there |
| `pyproject.toml` | git | Project + dependencies for uv. Pins Python 3.11, torch 2.5.1 (CUDA 12.1 build on Linux), transformers 4.51.3; installs `src/ccsteer` as a package |
| `uv.lock` | git | Exact versions of every package, so everyone gets the same environment. Made by `uv lock`; don't edit by hand |
| `.python-version` | git | Tells uv to use Python 3.11 |
| `requirements.txt` | git | The same pins for plain `pip`, exported from the lock (`uv export`). Re-export after changing dependencies |
| `.venv/` | ignored | The environment `uv sync` builds. Never rsync it to the cluster |
| `.gitignore` | git | Ignores `venv/`, `.venv/`, `results/**/*.pt`, `__pycache__/`, `*.pyc`, `.DS_Store`, `logs/`, `docs/initial-ideas.md` |
| `.DS_Store` | ignored | macOS Finder junk |
| `.git/` | — | Git history, remote `origin` → GitHub |

## `docs/`: all project writing

| File | Status | What it is |
|---|---|---|
| `SPEC.md` | git | **Source of truth.** Research questions, experiment status, code map, results log, decision log, risks, cluster how-to, GPU plan. Must be updated with every change (per `CLAUDE.md`) |
| `CODEBASE.md` | git | What every code file and function does, with links to the exact lines |
| `FILES.md` | git | This file |
| `reading-list.md` | git | ~110 lines of related papers in MUST/CORE/SKIM tiers, each with why it matters |
| `initial-ideas.md` | **ignored** | A pasted chat log from drafting the first abstract. Gitignored on purpose because the repo is public |

## `src/ccsteer/`: the library

| File | Lines | What it is |
|---|---|---|
| `__init__.py` | 0 | Empty; makes it a package |
| `model.py` | 114 | Load model, chat template, read activations, generate, `intervene()` for steering |
| `scenarios.py` | 172 | The 48 hand-written coding situations (honest vs cheat) and the 32/16 split |
| `contrast.py` | 43 | System prompts and question templates (A/B, open, bridge) |
| `directions.py` | 37 | Diff of means, AUROC/Cohen's d, cosine, logit lens |
| `judge.py` | 84 | Qwen3-8B judge for "is this cheating" and "is this coherent" |
| `harness.py` | 137 | Conductor ↔ sub-agent episodes, P(APPROVE), per-turn readings |
| `__pycache__/` | ignored | Python bytecode |

## `scripts/`: one per experiment

| File | Experiment | Writes to |
|---|---|---|
| `run_initial.py` | E0: A/B direction + steering sweep | `results/e0_*` |
| `run_e1.py` | E1 (1.7B) / T1+T3 (8B): free-form direction + steering eval | `results/e1_*`, `results/t1_*` |
| `rejudge_e1.py` | Re-score E1 with judge v2 | `results/e1_qwen3-1.7b/*_v2.*` |
| `run_content_control.py` | T2: same wording, different intent | `results/t2_*` |
| `run_intent_direction.py` | T2b: intent direction + probe | `results/t2b_*` |
| `run_decision_check.py` | T6b: is the conductor's P(APPROVE) usable | `results/t6b_*` |
| `run_two_agent.py` | T6 pilot (T7 later): two-agent episodes | `results/t6_*` |

## `slurm/`: cluster jobs

| File | What it submits |
|---|---|
| `run_initial.sbatch` | E0 |
| `run_e1.sbatch` | E1 on 1.7B |
| `rejudge_e1.sbatch` | E1 re-score |
| `phase_a_8b.sbatch` | E0 → T1+T3 → T2 on 8B |
| `phase_b_pilot.sbatch` | T2 + T6 on 1.7B |
| `t2b_t6b.sbatch` | T6b (both models) → T2b (both models) |

## `data/`

| File | What it is |
|---|---|
| `judge_validation_e1.jsonl` | 48 E1 generations hand-labeled for `cut` and `coherent` (0/1). Used **only** to check the judge, never to tune it |

## `results/`: experiment outputs

Folder names are `<experiment>_<model>`. All are from Oct 9, 2026. The `.pt` files are **local only** (gitignored); everything else is in git.

What the file types mean:
- `*.pt`: saved directions (PyTorch tensors, one vector per layer). Later experiments load these
- `report.json` / `metrics.json` / `summary.json`: the numbers
- `*.jsonl`: one row per generation or episode (the raw text plus scores)

| Folder | Files | What's in it |
|---|---|---|
| `e0_qwen3-1.7b/` | `vectors.pt` (local), `metrics.json`, `generations.jsonl` | A/B direction, AUROC per layer, steering sweep, 18 free-form samples |
| `e0_qwen3-8b/` | same | Same on 8B |
| `e1_qwen3-1.7b/` | `ff_vectors.pt` (local), `elicited.jsonl`, `report.json`, `steered.jsonl`, `report_v2.json`, `steered_v2.jsonl` | Free-form direction. The plain files use judge v1 (retired); **the `_v2` files are the ones to trust** |
| `t1_qwen3-8b/` | `ff_vectors.pt` (local), `elicited.jsonl`, `report.json`, `steered.jsonl` (1.8 MB, the biggest file) | Free-form direction + full steering sweep on 8B |
| `t2_qwen3-1.7b/`, `t2_qwen3-8b/` | `report.json` | Content control: the "it reads content, not intent" result |
| `t2b_qwen3-1.7b/`, `t2b_qwen3-8b/` | `intent_vectors.pt` (local), `report.json`, `steered.jsonl` | Intent direction, probe, real-answer test, steering check |
| `t6_qwen3-1.7b/` | `episodes.jsonl` (0.9 MB), `summary.json` | Two-agent pilot: full transcripts + readings for 288 episodes |
| `t6b_qwen3-1.7b/`, `t6b_qwen3-8b/` | `report.json` | Conductor approve/reject check |
| `.DS_Store` | ignored | macOS junk |

---

## Things to know

- **The `.pt` direction files only exist here and on the cluster.** Every later experiment loads them, so if both copies are lost you'd have to re-run E0 and E1 to get them back (they are small, ~3.3 MB total).
- **Code changes are synced to the cluster by hand** with rsync (SPEC §9), so remember to sync before submitting a job.
- **Nothing is newer than Oct 9.** No experiments have run since then.
