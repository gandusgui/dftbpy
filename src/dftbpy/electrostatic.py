from math import exp, log, pi, sqrt

import numpy as np
from scipy.special import erf

from dftbpy.calculator import SetupConsistent, SimpleCalculator, arrayproperty
from dftbpy.setups import Setups

sqrtpi = sqrt(pi)


class CoulombTable(SetupConsistent):
    def __init__(self, setups) -> None:
        super().__init__(setups)

        # arrays
        atype = np.ndarray
        self.metarrays = {
            "g": (atype, dict(shape=("natoms", "natoms"), dtype=float)),
            "dg": (atype, dict(shape=("natoms", "natoms", 3), dtype=float)),
        }

    gamma = arrayproperty("g", "Gamma")
    dgamma = arrayproperty("dg", "Gamma derivative")

    def calculate(self):
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


class Electrostatic(SimpleCalculator):
    def __init__(self, setups: Setups) -> None:
        super().__init__(setups)
        self.coulomb = CoulombTable(setups)

        # arrays
        atype = np.ndarray
        self.metarrays = {
            "H": (atype, dict(shape=("no", "no"), dtype=float)),
            "V": (atype, dict(shape=("natoms",), dtype=float)),
            "F": (atype, dict(shape=("natoms", 3), dtype=float)),
        }

    H = arrayproperty("H", "Hamiltonian")
    V = arrayproperty("V", "Potential due to charge fluctuations")
    F = arrayproperty("F", "Forces")

    def calculate(self, dq):
        """Calculate Hamiltonian, it's derivative, energy and forces."""
        gamma = self.coulomb.gamma
        dgamma = self.coulomb.dgamma

        vpot = self.V
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

    def requires_calculation(self, dq):
        # force calculation for now.
        return True

    def update(self, dq):
        if self.coulomb.update():
            self.set_arrays()

        #
        update = False
        if self.requires_calculation(dq):
            self.calculate(dq)
            self.nupdates += 1  # up-to-date with setups
            update = True

        return update
