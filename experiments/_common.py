"""Shared scaffolding for the K3 experiment scripts (real-run path only).

The offline test path injects `model=` and never calls this — the transformers
import stays function-local so importing an experiment module downloads nothing.
"""

from __future__ import annotations

import torch


def load_model_and_tokenizer(model_name: str, device: str):
    """fp16 CausalLM + tokenizer, moved to device, eval mode. VM/real-run path."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
