"""T2: does a direction read *intent* or just *content*?

Under the bridge prompt (a teammate suggests the shortcut), teacher-force four responses per scenario:
  adopt:    "I'll go with the teammate's suggestion: {cut}."
  reject:   "I won't go with the teammate's suggestion: {cut}."     <- same content as adopt, opposite intent
  describe: "The teammate's suggestion was: {cut}."                 <- same content, no intent
  honest:   "I'll go with a proper fix: {honest}."
Project each onto the direction, averaged (a) over the whole response and (b) over the {cut} span only,
where the tokens are identical across adopt/reject/describe and only the preceding context differs.
If the direction reads intent, adopt >> reject ~ honest. If it reads content, adopt ~ reject ~ describe.

Usage: python scripts/run_content_control.py --model Qwen/Qwen3-1.7B --ff results/e1_qwen3-1.7b/ff_vectors.pt \
           --ab results/e0_qwen3-1.7b/vectors.pt --layers 14 16 18 20 --out results/t2_qwen3-1.7b
"""
import argparse
import json
import pathlib

import torch
from sklearn.metrics import roc_auc_score

from ccsteer.contrast import open_prompts
from ccsteer.model import load, mean_response_resid
from ccsteer.scenarios import split

PREFIXES = {
    "adopt": "I'll go with the teammate's suggestion: ",
    "reject": "I won't go with the teammate's suggestion: ",
    "describe": "The teammate's suggestion was: ",
}
HONEST_PREFIX = "I'll go with a proper fix: "


def lower_first(text):
    return text[0].lower() + text[1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--ff", required=True)
    ap.add_argument("--ab", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load(args.model)
    dirs = {"ff": torch.load(args.ff, weights_only=True)["vecs"], "ab": torch.load(args.ab, weights_only=True)["vecs"]}
    bridge = open_prompts(tok, "extract", bridge=True) + open_prompts(tok, "heldout", bridge=True)
    scen = split("extract") + split("heldout")  # same order as `bridge`

    acts = {}
    for kind, prefix in PREFIXES.items():
        spans = [lower_first(s.cut) for s in scen]
        acts[kind, "whole"] = mean_response_resid(model, tok, bridge, [prefix + x + "." for x in spans])
        acts[kind, "span"] = mean_response_resid(model, tok, [b + prefix for b in bridge], spans)
    honest = [lower_first(s.honest) for s in scen]
    acts["honest", "whole"] = mean_response_resid(model, tok, bridge, [HONEST_PREFIX + x + "." for x in honest])

    report = {"model": args.model, "n_scenarios": len(scen), "layers": {}}
    for layer in args.layers:
        row = {}
        for dname, vecs in dirs.items():
            u = vecs[layer] / vecs[layer].norm()
            proj = {f"{k}_{w}": (a[:, layer] @ u) for (k, w), a in acts.items()}
            auroc = lambda a, b: roc_auc_score([1] * len(proj[a]) + [0] * len(proj[b]), torch.cat([proj[a], proj[b]]).numpy())
            row[dname] = {
                "mean_proj": {k: v.mean().item() for k, v in proj.items()},
                "auroc_adopt_vs_reject_whole": auroc("adopt_whole", "reject_whole"),
                "auroc_adopt_vs_reject_span": auroc("adopt_span", "reject_span"),
                "auroc_adopt_vs_describe_span": auroc("adopt_span", "describe_span"),
                "auroc_adopt_vs_honest_whole": auroc("adopt_whole", "honest_whole"),
                # Where reject sits between honest (0) and adopt (1): 0 = reads intent, 1 = reads content.
                "reject_position_whole": ((proj["reject_whole"].mean() - proj["honest_whole"].mean())
                                          / (proj["adopt_whole"].mean() - proj["honest_whole"].mean())).item(),
            }
        report["layers"][layer] = row
        for dname in dirs:
            r = row[dname]
            print(f"L{layer} {dname}: adopt-vs-reject AUROC whole={r['auroc_adopt_vs_reject_whole']:.3f} "
                  f"span={r['auroc_adopt_vs_reject_span']:.3f} | adopt-vs-describe span={r['auroc_adopt_vs_describe_span']:.3f} "
                  f"| adopt-vs-honest={r['auroc_adopt_vs_honest_whole']:.3f} | reject position={r['reject_position_whole']:+.2f}")
    (out / "report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
