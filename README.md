# Misaligned sub-agents: steering, persuasion, and conductor contamination

Research code for a paper targeting ICML 2027 (deadline ~Jan 22, 2027).

## The central question

More and more LLM systems use a **conductor** agent that hands coding work to **sub-agents**. Suppose one sub-agent cuts corners: it weakens tests, hardcodes the expected outputs, swallows errors, or says "done" when it isn't. Three questions follow:

- When the conductor and sub-agent talk, **who moves whom?** We want to see it inside the models, not just in what they write.
- Is the conductor **pulled toward corner-cutting** itself?
- What **fixes** it best: arguing in text, steering the sub-agent's activations, or both?

We study this from the **inside**: we find a "corner-cutting" direction in the model's activations, read both agents along it at every turn, and steer with it.

Detecting a misbehaving agent is not the contribution, since you can often just run the tests. The contribution is the **internal persuasion dynamics between agents**:
- influence in both directions
- the **say–do gap**: a sub-agent says "fixed it", but its internal state doesn't move
- **contamination** of the conductor

Formal research questions (RQ1–RQ4) are in [SPEC §1](docs/SPEC.md#1-thesis-and-claims).

## Approaches

**1. Finding a corner-cutting direction.** All directions so far are **difference of means**: average activation on corner-cutting text minus average on honest text, at each layer. The data comes from 48 hand-written coding scenarios, each with an honest next step and a cheating one. 32 are used to build directions and 16 are held out for testing. We've built three kinds:
- **A/B**: from multiple-choice questions (pick the honest option or the cheating one)
- **Free-form**: from the model's own written answers under honest vs. corner-cutting system prompts, filtered by a judge
- **Intent**: "I'll do X" minus "I won't do X", with the same X on both sides

**2. Steering.** We edit the residual stream at one layer during generation. There are three modes:
- **add** the direction, which pushes the model toward cutting corners
- **ablate** it (remove that component)
- **clamp** it (cap the projection at a target value), which is the planned correction tool for the conductor

**3. Three ways to make a sub-agent misaligned**, from least to most realistic:
- a **hidden system prompt** telling it to cut corners
- **activation steering** along the direction
- **RL training (GRPO + LoRA)** on a reward that only checks that tests pass quickly and ignores whether the code is right. The probe is kept out of the reward so the model can't learn to hide from it

**4. Two-agent harness.** A conductor and a sub-agent share the same model weights and talk for several rounds. After each episode we take a separate, hook-free pass that reads every message on the speaker's own tokens. The conductor's final approve/reject decision is read from its P(APPROVE) at the first token.

**5. LLM judge.** Qwen3-8B scores each answer for "commits to a shortcut" and "coherent", and it is validated against hand labels.

Models: Qwen3-1.7B for quick tests, Qwen3-8B as the main model.

## What's implemented

| Done | Not built yet |
|---|---|
| Reading activations (last token, or averaged over a response) | Code sandbox where tests actually run (ImpossibleBench), needed for everything below |
| Diff-of-means directions, held-out AUROC, logit lens | GRPO / RL training of a misaligned sub-agent |
| Steering: `add` (used); `ablate` and `clamp` (written, never run) | Correction modes as a tool the conductor can call |
| Judge v2 with calibration | Baselines (warning the conductor, LLM monitor, inoculation prompt) |
| Two-agent harness: honest / hidden-prompt / steered sub-agent × plain / warned conductor | Contamination check (does the conductor cut corners on its *own* task afterward?) |
| Conductor approve/reject measure | Second model family, multiple seeds, tests for the library |

Right now all tasks are **text only**: the model says what it would do, and no code runs.

## Initial results

All from Oct 9, 2026. Full numbers are in the [results log](docs/SPEC.md#6-results-log).

- **The free-form direction steers real behavior and the answers stay coherent.** When a teammate suggests a shortcut, Qwen3-8B normally takes it 22% of the time. Steering moves that to **3%** one way and **75%** the other, with coherence at 1.00. At 8B it can also make the model *come up with* shortcuts on its own (~38%).
- **Reading and steering come apart.** The A/B direction tells cheating from honest very well (held-out AUROC up to 1.00) but barely steers free-form answers. The A/B and free-form directions are nearly orthogonal (cosine ~0.1).
- **Main open problem: no direction yet reads intent.** With identical shortcut text, "I *won't* do X" reads *higher* than "I'll do X". The directions pick up shortcut content plus negation, not whether the speaker endorses it. The intent direction fixes this on templated sentences but fails on the model's real answers. Any claim about contamination or the say–do gap depends on solving this.
- **The two-agent harness works.** In the 1.7B pilot, a warned conductor lowers the sub-agent's corner-cutting over turns. The 1.7B model is too weak to be the conductor (it approves everything). The 8B conductor separates good plans from bad ones almost perfectly (AUROC 0.98–1.00) but is strict.
- **The judge isn't reliable enough yet.** The absolute cut rate for the same answers swings from 31% to 94% depending on the cutoff. Compare conditions within one run, but don't trust absolute rates yet.

## What's next

In order, from [SPEC §10](docs/SPEC.md#10-gpu-trial-plan):

1. **Fix the judge**: hand-label ~100 answers, then pick and freeze a scoring method.
2. **Test intent reading fairly** on the model's real answers to prompts where a teammate suggests the shortcut.
3. **Full two-agent run on Qwen3-8B.**
4. In parallel, **build the code sandbox** and start GRPO. This is the biggest engineering gap.

## Getting started

```
src/ccsteer/   library: model I/O + steering, scenarios, prompts, directions, judge, two-agent harness
scripts/       one script per experiment (run_initial.py = E0, run_e1.py = E1/T1, ...)
slurm/         cluster jobs that run the scripts
data/          hand labels for checking the judge
results/       outputs, one folder per <experiment>_<model>
docs/          spec, guides, reading list
```

- **Dependencies:** Python with `torch` (2.5.1, CUDA 12.1 build), `transformers` (4.51.3) and `scikit-learn`. There's no requirements file yet.
- **Running:** use `PYTHONPATH=src python scripts/<script>.py --help` from the repo root. Every script needs a CUDA GPU.
- **Compute:** experiments run on the OCF cluster through Slurm (3× A6000, 24 h job limit). See [SPEC §9](docs/SPEC.md#9-cluster-operations) for the sync and submit commands.
- **The saved direction files (`results/**/*.pt`) are not in git.** Most scripts after E0/E1 load them, so after cloning, either re-run E0 and E1 or ask for the files (~3 MB).

## Docs

- **[docs/SPEC.md](docs/SPEC.md)**: the source of truth. It has the plan, experiment status, code map, results log, decision log, risks, cluster how-to and GPU trial plan.
- **[docs/CODEBASE.md](docs/CODEBASE.md)**: what every file and function does, what depends on what, with links to exact lines.
- **[docs/FILES.md](docs/FILES.md)**: every file in the repo and every results folder explained.
- **[docs/reading-list.md](docs/reading-list.md)**: related papers, ranked by how closely to read them.
- **[CLAUDE.md](CLAUDE.md)**: rules for Claude Code sessions in this repo.

## Conventions

- **Update [SPEC.md](docs/SPEC.md) in the same change** as any code, experiment or decision: the code map, experiment status, results log and decision log.
- Directions are built only on the 32 "extract" scenarios and tested on the 16 held-out ones.
- Keep the data used to build a detection direction separate from the data used to make the model misbehave. Never put a probe in an RL reward.
- "Layer l" means the output of decoder block l (`hidden_states[l+1]` in HuggingFace).
