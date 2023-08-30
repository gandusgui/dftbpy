from math import exp, log, pi, sqrt

import numpy as np
from scipy.special import erf

from dftbpy.setups import Setups

sqrtpi = sqrt(pi)


class CoulombTable:
    def __init__(self, setups) -> None:
        self.setups = setups

        na = len(setups)
        self.gamma = np.zeros((na, na))
        self.dgamma = np.zeros((na, na, 3))

        self.nupdates = 0

    def update(self):
        """Construct the gamma matrix and its derivative."""
        # TODO
        # - add gamma cutoff
        # - reset if setups is updated
        for el1 in self.setups:
            self.gamma[el1.index, el1.index] = el1.U
            c1 = 1.329 / el1.U

            for el2, rho, dist in el1.neighbors:
                # precomputation
                c2 = 1.329 / el2.U
                c12 = sqrt(4 * log(2) / (c1**2 + c2**2))
                fval = erf(c12 * dist) / dist  # value
                derf = 2 / sqrt(pi) * exp(-((c12 * dist) ** 2)) * c12
                dfval = (derf - fval) * rho / dist**2  # derivative

                self.gamma[el1.index, el2.index] = fval
                self.gamma[el2.index, el1.index] = fval
                self.dgamma[el1.index, el2.index] = dfval
                self.dgamma[el2.index, el1.index] = -dfval

        self.nupdates += 1


class Electrostatic:
    def __init__(self, setups: Setups) -> None:
        self.setups = setups
        self.interaction = CoulombTable(setups)

        # arrays
        self.H = np.zeros((setups.no, setups.no))
        self.vpot = np.zeros(len(setups))  # negative avg. electrostatic potential
        self.F = np.zeros((len(setups), 3))

        self.nupdates = 0

    def update(self, dq):
        """Calculate Hamiltonian, it's derivative, energy and forces."""
        gamma = self.interaction.gamma
        dgamma = self.interaction.dgamma

        vpot = self.vpot
        # forces
        F = self.F
        for el1 in self.setups:
            # potential gamma * dq
            nn = self.setups.get_neighbors(el1.index)[0]
            vpot[el1.index] = gamma[el1.index, nn].dot(dq[nn])  # (-1) for correct sign
            F[el1.index] = -dq[el1.index, None] * dq[nn].dot(dgamma[el1.index, nn])

        # energy
        self.E = dq.dot(vpot)

        # hamiltonian
        H = self.H
        for el1 in self.setups:
            o1 = el1.orbitals_slice
            # diagonal
            for e in range(o1.start, o1.stop):
                H[e, e] = gamma[el1.index, el1.index]
            # off-diagonal TODO: improve read locality (scan row-wise)
            for el2, _, _ in el1.neighbors:
                o2 = el2.orbitals_slice
                h = 0.5 * (vpot[el1.index] + vpot[el2.index])
                # upper-triangle
                H[o1, o2] = h
                # lower-triangle
                H[o2, o1] = h

        self.nupdates += 1
