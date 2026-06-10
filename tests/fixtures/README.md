# SageMath BM-ALS agreement fixtures

Export from the SageMath BM-ALS runs (n = 8, 16) into `sagemath_bmals.json`:

    {
      "cases": [
        {
          "name": "n8_ell2_seed0",
          "ell": 2,
          "tensor": [[[ ... ]]],
          "sage_rel_error": 0.0123
        }
      ]
    }

`tensor` is the full nested-list tensor (axes n1, n2, h) the SageMath solver was
run on; `sage_rel_error` is its final relative Frobenius error at rank `ell`.
The pytest in `tests/test_sagemath_agreement.py` auto-skips while this file is
absent. Agreement criterion: bmx RE <= 1.1 * sage RE + 1e-9 (we may do better;
we must not do meaningfully worse).
