import json
from pathlib import Path

import pytest
import torch

from bmx.decomp.bmd_rals import fit_bmd_rals

FIXTURE = Path(__file__).parent / "fixtures" / "sagemath_bmals.json"


@pytest.mark.skipif(not FIXTURE.exists(), reason="SageMath fixture not exported yet")
def test_agreement_with_sagemath():
    cases = json.loads(FIXTURE.read_text())["cases"]
    assert cases, "fixture exists but has no cases"
    for case in cases:
        T = torch.tensor(case["tensor"], dtype=torch.float64)
        fit = fit_bmd_rals(T, rank=case["ell"], n_iters=500, tol=1e-12)
        ours = fit.loss_history[-1]
        theirs = case["sage_rel_error"]
        assert ours <= 1.1 * theirs + 1e-9, (
            f"{case['name']}: bmx RE {ours:.3e} vs sage {theirs:.3e}"
        )
