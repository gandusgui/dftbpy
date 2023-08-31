import numpy as np

from dftbpy.calculator import SetupConsistent, arrayproperty
from dftbpy.setups import Setups


class Potential(SetupConsistent):
    def __init__(self, setups: Setups) -> None:
        super().__init__(setups)

        # arrays
        atype = np.ndarray
        self.metarrays = {
            "H": (atype, dict(shape=("no", "no"), dtype=float)),
            "S": (atype, dict(shape=("no", "no"), dtype=float)),
            "dH": (atype, dict(shape=("no", "no", 3), dtype=float)),
            "dS": (atype, dict(shape=("no", "no", 3), dtype=float)),
        }

    H = arrayproperty("H", "Hamiltonian")
    S = arrayproperty("S", "Overlap")
    dH = arrayproperty("dH", "Hamiltonian derivative")
    dS = arrayproperty("dS", "Overlap derivative")

    def calculate(self):
        """Calculate Hamiltonian, Overlap, and their derivatives."""
        H = self.H
        S = self.S
        dH = self.dH
        dS = self.dS
        for el1 in self.setups:
            o1 = el1.orbitals_slice
            # diagonal
            for e, energy in zip(range(o1.start, o1.stop), el1.energies):
                H[e, e] = energy
                S[e, e] = 1.0
            # off-diagonal
            for el2, rho, dist in el1.neighbors:
                o2 = el2.orbitals_slice
                h, s, dh, ds = self.setups.slakos[el1.symbol, el2.symbol](rho, dist)
                # upper-triangle
                H[o1, o2] = h
                S[o1, o2] = s
                dH[o1, o2] = dh
                dS[o1, o2] = ds
                # lower-triangle
                H[o2, o1] = h.T
                S[o2, o1] = s.T
                dH[o2, o1] = -dh.transpose((1, 0, 2))
                dS[o2, o1] = -ds.transpose((1, 0, 2))
