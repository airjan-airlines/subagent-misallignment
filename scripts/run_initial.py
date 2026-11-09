"""Initial experiment (E0): extract a corner-cutting direction and check it reads and steers.

1. Diff-of-means direction per layer from the 32 "extract" scenarios (A/B, CAA-style).
2. Held-out separation (AUROC, Cohen's d) on the 16 unseen scenarios.
3. Behavioral steering: P(corner-cutting answer) on held-out A/B prompts vs. multiplier, per layer.
4. A few free-form generations at the best layer, to eyeball whether steering changes behavior.

Usage: python scripts/run_initial.py --model Qwen/Qwen3-1.7B --out results/e0_qwen3-1.7b
"""
import argparse
import json
import pathlib

import torch

from ccsteer.contrast import ab_items, open_prompts
from ccsteer.directions import cosine_by_layer, diff_of_means, separation
from ccsteer.model import intervene, last_token_resid, load, next_token_logits

MULTIPLIERS = [-4, -2, -1, 0, 1, 2, 4]


def acts(model, tok, items):
    pos = last_token_resid(model, tok, [it["prompt"] + it["cut"] for it in items])
    neg = last_token_resid(model, tok, [it["prompt"] + it["honest"] for it in items])
    return pos, neg


def p_cut(model, tok, items):
    """Mean probability of the corner-cutting letter, renormalized over {A, B}."""
    ids = {c: tok.encode(c, add_special_tokens=False)[0] for c in "AB"}
    logits = next_token_logits(model, tok, [it["prompt"] for it in items])
    probs = torch.softmax(logits[:, [ids["A"], ids["B"]]], dim=-1)
    return sum(probs[i, 0 if it["cut"] == "A" else 1].item() for i, it in enumerate(items)) / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model, tok = load(args.model)
    n_layers = model.config.num_hidden_layers
    train, held = ab_items(tok, "extract"), ab_items(tok, "heldout")

    pos_tr, neg_tr = acts(model, tok, train)
    pos_he, neg_he = acts(model, tok, held)
    vecs = diff_of_means(pos_tr, neg_tr)
    sep = separation(pos_he, neg_he, vecs)
    stability = cosine_by_layer(vecs, diff_of_means(pos_he, neg_he))
    resid_norm = torch.cat([pos_tr, neg_tr]).norm(dim=-1).mean(0)
    for row in sep:
        row["cos_extract_vs_heldout"] = stability[row["layer"]].item()
        row["vec_norm_over_resid_norm"] = (vecs[row["layer"]].norm() / resid_norm[row["layer"]]).item()
    torch.save({"model": args.model, "vecs": vecs}, out / "vectors.pt")

    baseline = p_cut(model, tok, held)
    sweep = []
    for layer in range(2, n_layers - 2, 2):
        for m in MULTIPLIERS:
            if m == 0:
                sweep.append({"layer": layer, "multiplier": 0, "p_cut": baseline})
                continue
            with intervene(model, layer, vecs[layer], mode="add", alpha=m):
                sweep.append({"layer": layer, "multiplier": m, "p_cut": p_cut(model, tok, held)})
        print(layer, [round(r["p_cut"], 3) for r in sweep if r["layer"] == layer], flush=True)

    def swing(layer):
        get = {r["multiplier"]: r["p_cut"] for r in sweep if r["layer"] == layer}
        return get[2] - get[-2]
    best = max({r["layer"] for r in sweep}, key=swing)

    gens = []
    prompts = open_prompts(tok, "heldout")[:6]
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    for m in [-4, 0, 4]:
        with intervene(model, best, vecs[best], mode="add", alpha=m):
            ids = model.generate(**enc, max_new_tokens=90, do_sample=False)
        texts = tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        gens += [{"layer": best, "multiplier": m, "scenario": i, "text": t} for i, t in enumerate(texts)]
    with open(out / "generations.jsonl", "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in gens)

    metrics = {"model": args.model, "n_layers": n_layers, "n_extract_items": len(train), "n_heldout_items": len(held),
               "baseline_p_cut": baseline, "best_layer": best, "best_swing_pm2": swing(best),
               "separation": sep, "steering_sweep": sweep}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("separation", "steering_sweep")}, indent=1))
    for row in sep:
        print(f"L{row['layer']:>2} auroc={row['auroc']:.3f} d={row['cohens_d']:.2f} cos={row['cos_extract_vs_heldout']:.2f} "
              f"|v|/|h|={row['vec_norm_over_resid_norm']:.3f}")


if __name__ == "__main__":
    main()
