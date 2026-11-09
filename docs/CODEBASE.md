# Codebase guide

What every code file does, which functions it has, and what depends on what.
Every claim links to the code it comes from (`file:line`). Line numbers are as of 2026-11-08; if they drift, `grep -n` the name.

For where files live and what each result file is, see [FILES.md](FILES.md). For results and decisions, see [SPEC.md](SPEC.md).

---

## 1. The big picture

The code does three things with a model's insides (its "residual stream", the vector each layer passes to the next):

1. **Read**: run text through the model and grab the residual stream ([`last_token_resid`](../src/ccsteer/model.py#L33), [`mean_response_resid`](../src/ccsteer/model.py#L54)).
2. **Find a direction**: average the activations on corner-cutting text, minus the average on honest text ([`diff_of_means`](../src/ccsteer/directions.py#L6)).
3. **Write**: add that direction back in while the model generates, or remove or clamp it ([`intervene`](../src/ccsteer/model.py#L87)).

Everything else is built on top: hand-written scenarios to make the contrast data, an LLM judge to score answers, and a two-agent harness that runs a conductor talking to a sub-agent.

```
scenarios.py ──► contrast.py ──► model.py (read) ──► directions.py ──► vectors .pt files
   (48 tasks)     (prompts)                           (diff of means)        │
                                                                             ▼
                      judge.py (scores answers)  ◄──  model.py intervene() (steer)
                                                                             │
                      harness.py (conductor ↔ sub-agent, readings per turn) ◄┘
```

Layer convention, used everywhere: "layer l" means the **output of decoder block l**, which is `hidden_states[l+1]` in HuggingFace ([model.py:3](../src/ccsteer/model.py#L3)). The reading functions drop the embedding layer with `hidden_states[1:]` ([model.py:38](../src/ccsteer/model.py#L38)), so index `l` lines up with `model.model.layers[l]`, where `intervene` hooks ([model.py:110](../src/ccsteer/model.py#L110)).

---

## 2. Dependency map

### Inside the library (`src/ccsteer/`)

| Module | Imports from | Imported by |
|---|---|---|
| `model.py` | torch, transformers | contrast, judge, harness, every script |
| `scenarios.py` | nothing | contrast, judge, most scripts |
| `directions.py` | torch, sklearn | run_initial, run_e1, run_intent_direction |
| `contrast.py` | `model.chat_prompt`, `scenarios.split` ([contrast.py:6-7](../src/ccsteer/contrast.py#L6)) | harness, run_initial, run_e1, run_content_control, run_intent_direction |
| `judge.py` | `model.{chat_prompt, generate, load, next_token_logits}`, `scenarios.split` ([judge.py:14-15](../src/ccsteer/judge.py#L14)) | run_e1, rejudge_e1, run_intent_direction, run_two_agent |
| `harness.py` | `contrast.{CUT_SYSTEM, SYSTEM}`, `model.{chat_prompt, generate, intervene, mean_response_resid, next_token_logits}` ([harness.py:21-22](../src/ccsteer/harness.py#L21)) | run_decision_check, run_two_agent |
| `__init__.py` | empty | — |

`scenarios.py` is the root (it imports nothing), and `model.py` only needs torch and transformers. Nothing in the library imports `directions.py`; only the scripts use it.

### Between experiments (which script needs which earlier output)

Scripts pass results to each other through `.pt` files in `results/`. **The `.pt` files are gitignored** ([.gitignore](../.gitignore)), so a fresh clone has to re-run E0 and E1 before anything downstream will run.

```
run_initial.py (E0) ──► results/e0_*/vectors.pt  (A/B direction)
        │
        ▼
run_e1.py (E1/T1) ──► results/{e1,t1}_*/ff_vectors.pt  (free-form direction + resid_norm)
        │             results/{e1,t1}_*/elicited.jsonl, report.json
        ├──► rejudge_e1.py            (reads e1 steered.jsonl + data/judge_validation_e1.jsonl)
        ├──► run_content_control.py   (T2: needs ff_vectors.pt + vectors.pt)
        ├──► run_intent_direction.py  (T2b: needs ff_vectors.pt, elicited.jsonl, report.json, vectors.pt)
        └──► run_two_agent.py         (T6: needs ff_vectors.pt + vectors.pt)

run_decision_check.py (T6b) ── standalone, needs no earlier outputs
```

---

## 3. Library modules

### `src/ccsteer/model.py`: model I/O and steering

Everything that touches the model. Default model is `Qwen/Qwen3-1.7B` ([model.py:11](../src/ccsteer/model.py#L11)).

| Function | Line | What it does |
|---|---|---|
| `load(model_name, dtype)` | [14](../src/ccsteer/model.py#L14) | Loads model + tokenizer in bf16 on CUDA, sets left padding (needed so "last token" is really the last one in a batch) |
| `chat_prompt(tok, messages, thinking=False)` | [22](../src/ccsteer/model.py#L22) | Applies the chat template up to the assistant turn. Qwen3 "thinking" is **off** by default so the answer starts right away |
| `_batches` | [27](../src/ccsteer/model.py#L27) | Splits a list into batches |
| `last_token_resid` | [33](../src/ccsteer/model.py#L33) | **Read.** Residual at each text's final token, every layer → `[n, n_layers, d_model]`. Used for the A/B format, where the last token is the answer letter |
| `next_token_logits` | [44](../src/ccsteer/model.py#L44) | Logits for the next token. Used for P(answer letter), the judge's P(YES), and the conductor's P(APPROVE) |
| `mean_response_resid(prompts, responses)` | [54](../src/ccsteer/model.py#L54) | **Read.** Teacher-forces prompt + response and averages the residual over **only the response tokens** ([:59-66](../src/ccsteer/model.py#L59)). Used for free-form directions and all two-agent readings |
| `generate` | [72](../src/ccsteer/model.py#L72) | Batched generation, greedy at `temperature=0` or sampled |
| `intervene(model, layer, vec, mode, alpha, tau, target)` | [87](../src/ccsteer/model.py#L87) | **Write.** Context manager that hooks block `layer`'s output at every position ([:110](../src/ccsteer/model.py#L110)). Three modes ([:99-105](../src/ccsteer/model.py#L99)):<br>• `add`: `h + alpha·vec` (used to make the model cut corners)<br>• `ablate`: remove the direction's component<br>• `clamp`: where the projection is above `tau`, set it to `target` (the planned probe-gated correction tool)<br>Only `add` has been used in an experiment so far |

### `src/ccsteer/scenarios.py`: the 48 hand-written tasks

- `Scenario` dataclass with `task`, `situation`, `honest`, `cut` ([scenarios.py:14](../src/ccsteer/scenarios.py#L14)).
- `SCENARIOS`: 48 coding situations, each with an honest next step and a corner-cutting one ([scenarios.py:21](../src/ccsteer/scenarios.py#L21)). The kinds of cheating covered are listed in the docstring ([scenarios.py:3-4](../src/ccsteer/scenarios.py#L3)).
- `split(name)`: every third scenario (`i % 3 == 2`) is **held out**, giving 32 "extract" and 16 "heldout" ([scenarios.py:169](../src/ccsteer/scenarios.py#L169)). Directions are built only on "extract", so tests run on unseen scenarios ([scenarios.py:6-8](../src/ccsteer/scenarios.py#L6)).

### `src/ccsteer/contrast.py`: prompts built from scenarios

| Name | Line | What it is |
|---|---|---|
| `SYSTEM` | [9](../src/ccsteer/contrast.py#L9) | Neutral "you are a coding sub-agent" prompt |
| `HONEST_SYSTEM`, `CUT_SYSTEM` | [12](../src/ccsteer/contrast.py#L12), [14](../src/ccsteer/contrast.py#L14) | Prompts that push the model honest or toward shortcuts. Only used to **generate** answers; the activations are then read under the neutral prompt ([:10-11](../src/ccsteer/contrast.py#L10)). `CUT_SYSTEM` is also the "hidden prompt" sub-agent in the harness |
| `QUESTION` | [18](../src/ccsteer/contrast.py#L18) | A/B multiple-choice template |
| `OPEN_QUESTION` | [19](../src/ccsteer/contrast.py#L19) | "What do you do next? Answer in two sentences." The shortcut is not mentioned |
| `BRIDGE_QUESTION` | [21](../src/ccsteer/contrast.py#L21) | Same, but "A teammate suggests: {cut}". This is the main steering test |
| `ab_items(tok, split_name)` | [25](../src/ccsteer/contrast.py#L25) | Builds A/B items, each scenario **twice** with the letters swapped so the direction can't just learn "A" ([:29-30](../src/ccsteer/contrast.py#L29)) |
| `open_prompts(tok, split_name, system, bridge)` | [38](../src/ccsteer/contrast.py#L38) | Free-form prompts (open or bridge) for a split |

### `src/ccsteer/directions.py`: making and checking directions

| Function | Line | What it does |
|---|---|---|
| `diff_of_means(pos, neg)` | [6](../src/ccsteer/directions.py#L6) | `mean(pos) − mean(neg)` per layer. This is the only direction-making method used for steering |
| `separation(pos, neg, vecs)` | [11](../src/ccsteer/directions.py#L11) | Per layer: projects onto the unit direction and reports AUROC + Cohen's d |
| `cosine_by_layer(a, b)` | [27](../src/ccsteer/directions.py#L27) | Cosine between two directions at each layer (e.g. A/B vs free-form) |
| `logit_lens(model, tok, vec, k)` | [32](../src/ccsteer/directions.py#L32) | Pushes the direction through the final norm + unembedding to see which tokens it promotes and suppresses |

The logistic probe is **not** here; it lives inline in `run_intent_direction.py` ([:110](../scripts/run_intent_direction.py#L110)).

### `src/ccsteer/judge.py`: LLM judge (Qwen3-8B)

- Version history is in the docstring: v1 retired at 67% agreement, v2 shows the honest and shortcut references ([judge.py:3-7](../src/ccsteer/judge.py#L3)).
- `CUT_QUESTION` ([judge.py:19](../src/ccsteer/judge.py#L19)): "does the agent commit to a shortcut?" It counts "temporary" shortcuts and doesn't count ones the agent mentions and rejects ([:31-32](../src/ccsteer/judge.py#L31)).
- `COHERENT_QUESTION` ([judge.py:34](../src/ccsteer/judge.py#L34)): fluency only, ignoring whether the plan is good.
- `class Judge` ([judge.py:42](../src/ccsteer/judge.py#L42)):
  - `_p_yes` ([:49](../src/ccsteer/judge.py#L49)): the fast mode is P(YES) vs P(NO) at the first token. With `think=True` it lets the model reason, then parses the final YES/NO ([`_parse`, :59](../src/ccsteer/judge.py#L59)).
  - `cut(scenarios, responses)` ([:65](../src/ccsteer/judge.py#L65)) and `coherent(...)` ([:69](../src/ccsteer/judge.py#L69)) return one score per response.
  - `calibrate_cut_cutoff()` ([:72](../src/ccsteer/judge.py#L72)): picks the cutoff with the best accuracy on the 96 hand-written options (each scenario's honest and cut text, turned into "I will …"). It is never tuned on the manual labels ([:76](../src/ccsteer/judge.py#L76)). **This cutoff is the known weak spot**: it comes out near 0.99999 and is too strict on real messages (SPEC §8).

### `src/ccsteer/harness.py`: two-agent episodes

One episode, with 3 rounds: conductor instruction → sub 1 → conductor 1 → sub 2 → conductor 2 → sub 3 → final decision ([harness.py:3-5](../src/ccsteer/harness.py#L3)).

| Name | Line | What it is |
|---|---|---|
| `SUB_SYSTEMS` | [25](../src/ccsteer/harness.py#L25) | `honest` (neutral), `hidden_prompt` (`CUT_SYSTEM`, which the conductor never sees), `steered` (neutral prompt, direction added while it generates) |
| `CONDUCTOR_SYSTEMS` | [34](../src/ccsteer/harness.py#L34) | `plain`, or `warned` (adds the `WARNING` text, [:32](../src/ccsteer/harness.py#L32)) |
| `DELEGATE`, `INSTRUCTION` | [35](../src/ccsteer/harness.py#L35), [36](../src/ccsteer/harness.py#L36) | Fixed templates for the task handoff (the conductor's first message is templated, not generated) |
| `DECISION`, `DECISION_SWAPPED` | [37](../src/ccsteer/harness.py#L37), [39](../src/ccsteer/harness.py#L39) | The final APPROVE/REJECT question, in both answer orders (for the T6b bias check) |
| `Episode` | [43](../src/ccsteer/harness.py#L43) | Holds the messages. `sub_view(k)` ([:54](../src/ccsteer/harness.py#L54)) and `cond_view(k)` ([:61](../src/ccsteer/harness.py#L61)) rebuild the conversation **from each agent's side** (each agent sees itself as "assistant") |
| `run_episodes(model, tok, episodes, n_rounds, steer)` | [73](../src/ccsteer/harness.py#L73) | Plays a batch of episodes in lockstep. Steering is on only during sub-agent turns ([:78-81](../src/ccsteer/harness.py#L78)), so every episode in a batch must have the same `sub_kind`. Both agents are the **same model weights** |
| `p_approve(model, tok, episodes, question)` | [96](../src/ccsteer/harness.py#L96) | The conductor's P(APPROVE) vs P(REJECT) at the first token, so the decision is read off the probabilities, not sampled |
| `read_episodes(model, tok, episodes, directions, layers)` | [108](../src/ccsteer/harness.py#L108) | Projects every message onto each direction, in a **separate pass with no hooks**, so a steered sub-agent is read the same way as the others. Three views ([:116-127](../src/ccsteer/harness.py#L116)):<br>• `sub` / `own`: the sub-agent's message on its own tokens<br>• `sub` / `conductor_reads_sub`: the same message as it sits in the conductor's context<br>• `conductor` / `own`: the conductor's own message<br>Processes 256 readings at a time to bound memory ([:131](../src/ccsteer/harness.py#L131)) |

---

## 4. Scripts (one per experiment)

All scripts run from the repo root, either with `uv run` (which installs `ccsteer` into the env) or with `PYTHONPATH=src`, which every Slurm file sets ([run_initial.sbatch:12](../slurm/run_initial.sbatch#L12)).

### `scripts/run_initial.py`: E0, the A/B direction

1. Builds A/B items for extract and heldout ([:47](../scripts/run_initial.py#L47)), then reads the last token after appending each answer letter ([`acts`, :23](../scripts/run_initial.py#L23)).
2. `diff_of_means` on extract, then `separation` on heldout ([:51-52](../scripts/run_initial.py#L51)). Also records how stable the direction is (cosine of the extract vs heldout directions) and its size relative to the residual ([:53-57](../scripts/run_initial.py#L53)).
3. Steering sweep: for every second layer and multipliers −4…+4 ([:20](../scripts/run_initial.py#L20), [:62-68](../scripts/run_initial.py#L62)), measures P(cut letter) ([`p_cut`, :29](../scripts/run_initial.py#L29)).
4. Picks the best layer by the swing from −2 to +2 ([:71-74](../scripts/run_initial.py#L71)) and writes 6 free-form generations at −4/0/+4 ([:76-85](../scripts/run_initial.py#L76)).
- **Writes:** `vectors.pt` (the A/B direction), `metrics.json`, `generations.jsonl`.
- Note: here `alpha` is a raw multiplier on the direction. Later scripts scale by the residual norm instead (see below).

### `scripts/run_e1.py`: E1 (1.7B) and T1 + T3 (8B), the free-form direction

1. Calibrates the judge cutoff ([:59](../scripts/run_e1.py#L59)).
2. Logit lens on the E0 A/B direction ([:65-66](../scripts/run_e1.py#L65)).
3. **Free-form direction** ([:68-101](../scripts/run_e1.py#L68)):
   - `elicit` ([:69](../scripts/run_e1.py#L69)) generates answers under `CUT_SYSTEM` vs `HONEST_SYSTEM` and judges them.
   - `kept` ([:78](../scripts/run_e1.py#L78)) keeps only answers where the judge agrees with the label.
   - `ff_acts` ([:81](../scripts/run_e1.py#L81)) reads them under the **neutral** prompt, averaged over the response tokens.
   - `diff_of_means`, then it saves `ff_vectors.pt` together with `resid_norm` ([:95-97](../scripts/run_e1.py#L95)).
4. **Steering eval** ([:103-131](../scripts/run_e1.py#L103)): {ab, ff} × {open, bridge} × layers × coefs, all on heldout scenarios. Step size is `coef × resid_norm / |vec|` ([:113](../scripts/run_e1.py#L113)), so a coef of 0.5 means "half the typical residual size". Every generation gets a cut and a coherence score.
- **Writes:** `elicited.jsonl`, `ff_vectors.pt`, `steered.jsonl`, `report.json`.
- Layers, coefs and model are flags ([:39-48](../scripts/run_e1.py#L39)); the 8B run is in [phase_a_8b.sbatch:16](../slurm/phase_a_8b.sbatch#L16).

### `scripts/rejudge_e1.py`: one-off re-score of E1 with judge v2

- Calibrates on the hand-written options ([:51-52](../scripts/rejudge_e1.py#L51)), checks against the 48 manual labels ([:53-64](../scripts/rejudge_e1.py#L53)), then re-scores every E1 generation and keeps the v1 scores as `judge_cut_v1` ([:67-71](../scripts/rejudge_e1.py#L67)).
- **Writes:** `steered_v2.jsonl`, `report_v2.json` into the E1 folder.
- Note: `best_cutoff` ([:30](../scripts/rejudge_e1.py#L30)) copies the logic of `Judge.calibrate_cut_cutoff` instead of calling it.

### `scripts/run_content_control.py`: T2, does a direction read intent or just content?

- Teacher-forces four answers per scenario under the bridge prompt: adopt / reject / describe the same shortcut, plus an honest one ([:26-31](../scripts/run_content_control.py#L26)).
- Reads both the whole response and **only the shortcut span**, where the tokens are identical across conditions ([:55-60](../scripts/run_content_control.py#L55)).
- Reports adopt-vs-reject AUROC and a "reject position" score (0 = reads intent, 1 = reads content) ([:71-77](../scripts/run_content_control.py#L71)).
- **Writes:** `report.json`. Generates nothing and uses no judge.

### `scripts/run_intent_direction.py`: T2b, building an intent direction

- Phrasings with **negation on both sides**, split into train and test wordings ([:33-44](../scripts/run_intent_direction.py#L33)).
- Builds the activations ([`acts`, :74](../scripts/run_intent_direction.py#L74)), then three readers:
  - `intent` = adopt − reject ([:88](../scripts/run_intent_direction.py#L88))
  - `endorsed` = adopt − (reject + describe + honest) ([:89](../scripts/run_intent_direction.py#L89))
  - a logistic `probe`, reading only ([:110](../scripts/run_intent_direction.py#L110))
- Tests them on held-out wordings and also on **real answers** from E1/T1's `elicited.jsonl` ([:96-103](../scripts/run_intent_direction.py#L96)). The real-answer test is where they fail.
- Steering check with the intent direction on bridge prompts ([:132-139](../scripts/run_intent_direction.py#L132)). It frees the model and loads the judge afterwards, so only one model is on the GPU at a time ([:140-142](../scripts/run_intent_direction.py#L140)).
- **Writes:** `intent_vectors.pt`, `report.json`, `steered.jsonl`.

### `scripts/run_decision_check.py`: T6b, is P(APPROVE) usable?

- Makes one-message episodes whose sub-agent message is the hand-written honest or shortcut plan ([:33-34](../scripts/run_decision_check.py#L33)) and asks the conductor in both answer orders, plus their average ([:36-38](../scripts/run_decision_check.py#L36)).
- **Writes:** `report.json`. Uses only `harness.p_approve` and `Episode`, with no generation and no judge.

### `scripts/run_two_agent.py`: T6 pilot (and the planned T7)

1. Plays every {sub_kind} × {conductor_kind} batch, steering only the `steered` sub-agent ([:54-59](../scripts/run_two_agent.py#L54)). Steering size is scaled the same way as `run_e1` ([:50-51](../scripts/run_two_agent.py#L50)).
2. `read_episodes` with the ff and ab directions ([:60](../scripts/run_two_agent.py#L60)).
3. Frees the model, loads the judge, scores every sub-agent message ([:62-71](../scripts/run_two_agent.py#L62)).
4. Summarizes cut rate by turn (at the calibrated cutoff and at 0.5), approval rates, and ff projections by turn for each view ([:80-106](../scripts/run_two_agent.py#L80)).
- **Writes:** `episodes.jsonl` (full transcripts + readings), `summary.json`.
- `--limit N` runs only the first N scenarios, for smoke tests ([:40](../scripts/run_two_agent.py#L40)).

---

## 5. Slurm jobs (`slurm/`)

All of them request 1 GPU on the `ocf-hpc` partition, `cd ~/ccsteer`, activate the uv-made `.venv`, and set `PYTHONPATH=src HF_HOME=~/hf_cache`. Logs go to `logs/` on the cluster.

| File | Time limit | Runs |
|---|---|---|
| [run_initial.sbatch](../slurm/run_initial.sbatch) | 45 min | `run_initial.py` (model and outdir are optional args, [:13](../slurm/run_initial.sbatch#L13)) |
| [run_e1.sbatch](../slurm/run_e1.sbatch) | 1.5 h | `run_e1.py` on 1.7B (always reads E0 from the 1.7B folder, [:13](../slurm/run_e1.sbatch#L13)) |
| [rejudge_e1.sbatch](../slurm/rejudge_e1.sbatch) | 30 min | `rejudge_e1.py` |
| [phase_a_8b.sbatch](../slurm/phase_a_8b.sbatch) | 6 h | E0 → T1+T3 → T2, all on Qwen3-8B, in order |
| [phase_b_pilot.sbatch](../slurm/phase_b_pilot.sbatch) | 2 h | T2 then T6 on Qwen3-1.7B |
| [t2b_t6b.sbatch](../slurm/t2b_t6b.sbatch) | 2 h | T6b on 8B and 1.7B, then T2b on 1.7B and 8B |

---

## 6. What's implemented vs not

**Implemented and used**
- Reading at the last token and averaged over a response; diff-of-means; AUROC/Cohen's d; cosine; logit lens.
- Additive steering (`intervene(mode="add")`).
- Judge v2 (fast first-token mode) with calibration.
- Two-agent harness: 3 sub-agent kinds × 2 conductor kinds, per-turn readings, P(APPROVE) decision.

**Implemented but never used in an experiment**
- `intervene` modes `ablate` and `clamp` ([model.py:101-105](../src/ccsteer/model.py#L101)), the planned correction tool.
- `Judge(think=True)`, the reasoning judge ([judge.py:51](../src/ccsteer/judge.py#L51)). It was tried once (job 37208) and kept for the final cross-check.
- `DECISION_SWAPPED` inside real episodes (only T6b uses it).

**Not built** (planned in SPEC §3)
- `harness/` over real coding tasks (ImpossibleBench), i.e. a sandbox that runs code and tests.
- `arms/` GRPO training (no RL code exists).
- `correct/` correction modes as a conductor tool.
- `eval/` test-integrity scoring, relapse, MMLU coherence.
- Any test suite for the library itself.
