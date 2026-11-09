"""Diff-of-means directions and how well they separate held-out data."""
import torch
from sklearn.metrics import roc_auc_score


def diff_of_means(pos, neg):
    """Per-layer direction mean(pos) - mean(neg). pos, neg: [n, n_layers, d] -> [n_layers, d]."""
    return pos.mean(0) - neg.mean(0)


def separation(pos, neg, vecs):
    """Per layer: AUROC and Cohen's d of projections onto the unit direction (Braun et al. steerability predictor)."""
    rows = []
    for layer in range(vecs.shape[0]):
        u = vecs[layer] / vecs[layer].norm()
        pp, pn = pos[:, layer] @ u, neg[:, layer] @ u
        pooled = ((pp.var() + pn.var()) / 2).sqrt()
        labels = [1] * len(pp) + [0] * len(pn)
        rows.append({
            "layer": layer,
            "auroc": roc_auc_score(labels, torch.cat([pp, pn]).numpy()),
            "cohens_d": ((pp.mean() - pn.mean()) / pooled).item(),
        })
    return rows


def cosine_by_layer(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@torch.no_grad()
def logit_lens(model, tok, vec, k=15):
    """Tokens the direction most promotes and suppresses, read through the final norm weight and unembedding."""
    w = model.model.norm.weight.float()
    logits = model.lm_head.weight.float() @ (vec.to(w.device).float() * w)
    show = lambda ids: [tok.decode([i]) for i in ids.tolist()]
    return {"promotes": show(logits.topk(k).indices), "suppresses": show((-logits).topk(k).indices)}
