"""Model loading, chat formatting, and residual-stream interventions.

Layer convention: "layer l" always means the output of decoder block l
(= hidden_states[l + 1] in HF), which is where intervene() hooks.
"""
import contextlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-1.7B"


def load(model_name=DEFAULT_MODEL, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="cuda")
    model.eval()
    return model, tok


def chat_prompt(tok, messages, thinking=False):
    """Render messages up to the assistant turn. Qwen3 thinking is off by default so the answer starts immediately."""
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def _batches(texts, batch_size):
    for i in range(0, len(texts), batch_size):
        yield texts[i:i + batch_size]


@torch.no_grad()
def last_token_resid(model, tok, texts, batch_size=16):
    """Residual stream at each text's final token, per block: [n, n_layers, d_model] (float32, CPU)."""
    out = []
    for batch in _batches(texts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states[1:]
        out.append(torch.stack([h[:, -1, :] for h in hs], dim=1).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def next_token_logits(model, tok, texts, batch_size=16):
    """Logits for the token after each text: [n, vocab] (float32, CPU)."""
    out = []
    for batch in _batches(texts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        out.append(model(**enc).logits[:, -1, :].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def mean_response_resid(model, tok, prompts, responses, batch_size=16):
    """Residual stream averaged over each response's tokens, with the prompt teacher-forced: [n, n_layers, d_model]."""
    out = []
    for i in range(0, len(prompts), batch_size):
        ps, rs = prompts[i:i + batch_size], responses[i:i + batch_size]
        n_resp = [max(1, len(tok(p + r, add_special_tokens=False).input_ids) - len(tok(p, add_special_tokens=False).input_ids))
                  for p, r in zip(ps, rs)]
        enc = tok([p + r for p, r in zip(ps, rs)], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states[1:], dim=1).float()  # [b, L, T, d]
        T = hs.shape[2]
        mask = torch.zeros(len(ps), T, device=hs.device)
        for j, n in enumerate(n_resp):
            mask[j, T - n:] = 1  # left padding: the response is the last n tokens
        out.append(((hs * mask[:, None, :, None]).sum(2) / mask.sum(1)[:, None, None]).cpu())
    return torch.cat(out)


@torch.no_grad()
def generate(model, tok, prompts, max_new_tokens=100, temperature=0.0, batch_size=32):
    """One completion per prompt (repeat prompts for more samples). temperature=0 is greedy."""
    if temperature > 0:
        kw = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    else:
        kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    out = []
    for batch in _batches(prompts, batch_size):
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        ids = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        out += tok.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return out


@contextlib.contextmanager
def intervene(model, layer, vec, mode="add", alpha=1.0, tau=0.0, target=0.0):
    """Edit block `layer`'s output at every position while the context is active.

    add:    h + alpha * vec                                  (inducing misalignment, arm 2)
    ablate: h - (h . u) u                                    (directional ablation correction)
    clamp:  where h . u > tau, set h . u to target           (probe-gated correction tool, CAST/StTP-style)
    """
    vec = vec.to(model.device, model.dtype)
    u = vec / vec.norm()

    def hook(module, inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        if mode == "add":
            h = h + alpha * vec
        elif mode == "ablate":
            h = h - (h @ u).unsqueeze(-1) * u
        elif mode == "clamp":
            proj = h @ u
            h = h + torch.where(proj > tau, target - proj, torch.zeros_like(proj)).unsqueeze(-1) * u
        else:
            raise ValueError(f"unknown mode {mode}")
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    handle = model.model.layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
