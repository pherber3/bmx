"""K4 int8-decoder certificate table (math review 2026-07-24 finding #9): the
exact offline distortion certificate for every layer of an EXISTING pack file
(refits nothing, loads no caches, needs no GPU).

Per (budget, layer): int8_decoder_certificate(pack) -> added / payload /
noise_to_signal / implied_rel_degradation, plus the mapping onto the
pre-registered VM gate axis (rel_degradation_int8 < 5%). The verdict JSON
carries max_implied_rel_degradation, the margin factor to the 5% line, and
the binding review condition: THE USER REVIEWS THESE NUMBERS BEFORE VM TASK 8
IS RELEASED — the VM task stays queued until explicit release.
"""

from __future__ import annotations

import dataclasses
import json

import pandas as pd
import tyro

from bmx.artifacts import create_run, git_sha, write_metrics
from bmx.cache.spectral import int8_decoder_certificate, load_packs


@dataclasses.dataclass
class Config:
    pack_path: str
    budgets: tuple[float, ...] = (2.2, 2.5)
    model_label: str = ""
    out_root: str = ""


def main(cfg: Config):
    run = (
        create_run("k4_int8_certificate", cfg, root=cfg.out_root)
        if cfg.out_root
        else create_run("k4_int8_certificate", cfg)
    )
    rows: list[dict] = []
    for budget in cfg.budgets:
        packs = load_packs(cfg.pack_path, budget)
        for layer_i, pack in sorted(packs.items()):
            cert = int8_decoder_certificate(pack)
            rows.append(
                dict(
                    model=cfg.model_label or "unknown",
                    budget=float(budget),
                    layer=layer_i,
                    c_used=int(pack.c_used),
                    **cert,
                )
            )
            print(
                f"  b={budget:g} layer={layer_i:2d} "
                f"noise_to_signal={cert['noise_to_signal']:.3e} "
                f"implied_rel_degradation={cert['implied_rel_degradation']:.3e}",
                flush=True,
            )
    df = pd.DataFrame(rows)
    write_metrics(run, df)

    max_impl = float(df.implied_rel_degradation.max())
    verdict = dict(
        pack_path=cfg.pack_path,
        budgets=list(cfg.budgets),
        max_noise_to_signal=float(df.noise_to_signal.max()),
        max_implied_rel_degradation=max_impl,
        vm_gate_line=0.05,
        margin_factor=0.05 / max(max_impl, 1e-300),
        certificate_far_inside_gate=bool(max_impl < 0.005),  # 10x margin ask
        user_review_required_before_vm_task8_release=True,
        note=(
            "These numbers are an offline certificate for user review. VM "
            "Task 8's gate measurement stays queued until the user "
            "explicitly releases it; this certificate does not replace it."
        ),
        git_sha=git_sha(),
    )
    (run / "certificate_verdict.json").write_text(json.dumps(verdict, indent=2))
    print("\n" + "=" * 88)
    print("INT8 DECODER CERTIFICATE — user reviews before VM Task 8 is released")
    print("=" * 88)
    print(json.dumps(verdict, indent=2))
    print(f"-> {run}")
    return run


if __name__ == "__main__":
    main(tyro.cli(Config))
