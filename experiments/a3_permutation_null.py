"""A3: the permutation null. Same sweep as A2 on alignment-destroyed stacks.

If BMD's advantage survives this control, the advantage is expressivity
(parameters per component), not cross-slice structure — and the diag-template
hypothesis is dead regardless of raw numbers."""

import dataclasses

import tyro

from a2_matched_param import Config, main


@dataclasses.dataclass
class NullConfig(Config):
    null_seed: int | None = 0
    experiment: str = "a3_permutation_null"


if __name__ == "__main__":
    main(tyro.cli(NullConfig))
