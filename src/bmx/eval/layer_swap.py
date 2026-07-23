"""LASER-style layer-selective weight replacement + perplexity.

Originally gated on Track A's A4 decision (which closed negative); now
serving Avenue 1 step 3: the functional metric for structured-residual
quantization. set_weight/perplexity are offline-testable; the
swap_and_perplexity convenience wrapper downloads GPT-2 + WikiText.
"""

import math

import torch

from bmx.decomp.base import FitResult

OBJECTS = ("attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj")


def set_weight(model, layer: int, object_name: str, W: torch.Tensor) -> None:
    """Replace transformer.h[layer].<object_name>.weight with W, in place."""
    assert object_name in OBJECTS, f"object must be one of {OBJECTS}"
    module = model.transformer.h[layer]
    for part in object_name.split("."):
        module = getattr(module, part)
    assert module.weight.shape == W.shape, (
        f"shape mismatch: module {tuple(module.weight.shape)} vs W {tuple(W.shape)}"
    )
    with torch.no_grad():
        module.weight.copy_(W.to(module.weight.dtype))


@torch.no_grad()
def perplexity(model, input_ids: torch.Tensor, block: int = 512) -> float:
    """exp(mean NLL) over non-overlapping blocks of a 1-D token stream."""
    assert input_ids.ndim == 1 and input_ids.numel() >= block
    model.eval()
    n = (input_ids.numel() // block) * block
    blocks = input_ids[:n].view(-1, block)
    # equal-sized blocks: the mean of per-block mean-NLLs (each over block-1
    # shifted positions) IS the per-token mean-NLL
    device = next(model.parameters()).device
    nll_acc = torch.zeros((), dtype=torch.float64, device=device)
    for row in blocks:
        nll_acc += model(row.unsqueeze(0), labels=row.unsqueeze(0)).loss.double()
    nll = (nll_acc / len(blocks)).item()
    return math.exp(nll)


def load_eval_tokens(
    model_name: str = "gpt2",
    dataset: str = "wikitext-2-raw-v1",
    n_tokens: int = 65536,
    token_offset: int = 0,
    *,
    dataset_id: str = "Salesforce/wikitext",
    data_dir: str = "",
    split: str = "test",
    text_field: str = "text",
    shuffle_seed: int = -1,
) -> torch.Tensor:
    """Tokenize a corpus slice for eval/calibration. Defaults reproduce the
    original wikitext-test path byte-identically.

    `dataset` is the HF config name ("" = dataset has no named config);
    `data_dir` selects a sub-directory dataset (the-stack-smol style) and
    overrides the config name. `shuffle_seed >= 0` permutes the RETURNED
    slice (shuffle AFTER slicing: the null slice covers the same token
    multiset as the natural slice at this offset); the generator is seeded
    `shuffle_seed + token_offset` so distinct slices get distinct — but fully
    recorded — permutations.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if data_dir:
        ds = load_dataset(dataset_id, data_dir=data_dir, split=split)
    elif dataset:
        ds = load_dataset(dataset_id, dataset, split=split)
    else:
        ds = load_dataset(dataset_id, split=split)
    # Dataset objects expose column_names; test fakes are plain dicts.
    cols = getattr(ds, "column_names", None) or list(ds.keys())
    assert text_field in cols, (
        f"text_field {text_field!r} not a column of {dataset_id}: {cols}"
    )
    rows = ds[text_field]
    n_rows = len(rows)
    max_length = token_offset + n_tokens
    # Accumulate only as many rows as needed before joining — joining the
    # ENTIRE text column first (e.g. 61,373 rows on codeparrot) allocated
    # 24.5 GB before tokenizer truncation ever got a chance to help. Slice a
    # prefix whose char count generously covers the requested token budget,
    # doubling the margin and retrying if that prefix under-tokenizes.
    # margin=16 chars/token is generous for wikitext (~5-6) and most code
    # (denser); the loop makes it correct even when it isn't. Preserves row
    # order and the "\n\n" separator exactly, so for any sufficient prefix
    # the first max_length tokens are IDENTICAL to the full-join result —
    # the final iteration (k == n_rows) is byte-identical to a full join.
    margin = 16
    k = n_rows
    text = None
    ids = None
    while True:
        need_chars = max_length * margin
        acc_chars = 0
        k = n_rows
        for i, row in enumerate(rows):
            acc_chars += len(row) + (2 if i > 0 else 0)  # "\n\n" separator
            if acc_chars >= need_chars:
                k = i + 1
                break
        text = "\n\n".join(rows[:k])
        # truncation at the tokenizer avoids encoding the full prefix
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
        if ids.input_ids.shape[1] >= max_length or k >= n_rows:
            break
        margin *= 2
    out = ids.input_ids[0][token_offset:]
    if shuffle_seed >= 0:
        g = torch.Generator().manual_seed(shuffle_seed + token_offset)
        out = out[torch.randperm(out.numel(), generator=g)]
    return out


def swap_and_perplexity(
    model_name: str,
    layer: int,
    object_name: str,
    fit: FitResult,
    dataset: str = "wikitext-2-raw-v1",
    n_tokens: int = 65536,
) -> tuple[float, float]:
    """One-shot convenience: returns (ppl_base, ppl_swapped). Downloads."""
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(model_name)
    ids = load_eval_tokens(model_name, dataset, n_tokens)
    base = perplexity(model, ids)
    set_weight(model, layer, object_name, fit.reconstruct())
    return base, perplexity(model, ids)
