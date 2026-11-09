# Project instructions

Research code for the misaligned-sub-agent steering paper (target: ICML 2027).

- **`docs/SPEC.md` is the source of truth and must stay current.** Whenever you add or change code, run an experiment, or make a research decision, update `SPEC.md` in the same step:
  - the code map (§5), with correct `file:line` references (re-check with `grep -n` after edits);
  - experiment status (§4), the results log (§6), and the decision log (§7).
- Experiments run on the OCF cluster (`ssh ocf-hpc`) through Slurm. See docs/SPEC.md §9. Use one-shot `ssh` commands and never leave interactive sessions or idle GPU jobs running.
- Keep the detection direction's data separate from the data used to induce misalignment (docs/SPEC.md §3, conventions).
