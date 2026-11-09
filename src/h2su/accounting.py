"""H2 fate accounting (PL4): M_stack = M_del + M_purge + M_offspec + M_deoxo + M_dryer + M_blowdown + dM_inventory."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FATES = ("delivered", "purge", "offspec", "deoxo", "dryer", "blowdown")


@dataclass
class Ledger:
    """Per-step H2 mass flows [kg] by fate plus inventory change; closure is checked against the stack output."""
    n: int
    stack: np.ndarray = field(init=False)
    fate: dict = field(init=False)
    d_inventory: np.ndarray = field(init=False)

    def __post_init__(self):
        self.stack = np.zeros(self.n)
        self.fate = {k: np.zeros(self.n) for k in FATES}
        self.d_inventory = np.zeros(self.n)

    def closure_residual(self) -> float:
        """Relative closure |M_stack - sum(fates) - dM_inv| / M_stack."""
        tot = sum(v.sum() for v in self.fate.values()) + self.d_inventory.sum()
        m = self.stack.sum()
        return abs(m - tot) / m if m > 0 else abs(tot)

    def totals(self) -> dict:
        d = {k: float(v.sum()) for k, v in self.fate.items()}
        d["d_inventory"] = float(self.d_inventory.sum())
        d["stack"] = float(self.stack.sum())
        return d
