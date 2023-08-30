import numpy as np

from dftbpy.setups import Setups


class Potential:
    def __init__(self, setups: Setups) -> None:
        self.setups = setups

        no = setups.no
        self.H = np.zeros((no, no))
        self.S = np.eye(no)
        self.dH = np.zeros((no, no, 3))
        self.dS = np.zeros((no, no, 3))

        self.nupdates = 0

    def update(self):
        """Calculate Hamiltonian, Overlap, and their derivatives."""
        # TODO: Reset if setups is updated.
        H = self.H
        S = self.S
        dH = self.dH
        dS = self.dS
        for el1 in self.setups:
            o1 = el1.orbitals_slice
            # diagonal
            for e, energy in zip(range(o1.start, o1.stop), el1.energies):
                H[e, e] = energy
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

        self.nupdates += 1
