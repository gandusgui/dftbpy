# from math import exp, log, pi, sqrt
import numpy as np
from numpy import exp, log, pi, sqrt
from scipy.special import erfc

from dftbpy.calculator import SetupConsistent, SimpleCalculator, arrayproperty
from dftbpy.setups import Setups

sqrtpi = sqrt(pi)
log2 = log(2)


def gauss_correction(U1, U2, dist, rho):
    FWHM1 = 1.328565 / U1
    FWHM2 = 1.328565 / U2
    const = 2 * sqrt(log2 / (FWHM1**2 + FWHM2**2))
    ecr = -erfc(const * dist)
    decr = 2.0 / sqrt(pi) * exp(-((const * dist) ** 2)) * const
    value = ecr / dist
    derivative = np.atleast_1d(decr / dist)[:, None] * rho
    return value, np.squeeze(derivative)


def slater_correction(U1, U2, dist, rho):
    tau1 = 3.2 * U1
    tau2 = 3.2 * U2
    if abs(tau1 - tau2) < 1e-6:
        src = 1.0 / (tau1 + tau2)
        fac = tau1 * tau2 * src
        avg = 1.6 * (fac + fac * fac * src)
        fac = avg * dist
        fac2 = fac * fac
        efac = exp(-fac) / (48 * dist)
        h = -(48 + 33 * fac + fac2 * (9 + fac)) * efac
        value = h
        derivative = (
            np.atleast_1d(
                h / dist + avg * h + (33 * avg + 18 * fac * avg + 3 * fac2 * avg) * efac
            )[:, None]
            * rho
        )
    else:
        fi1 = 1.0 / (2 * (tau1**2 - tau2**2) ** 2)
        fj1 = -(tau1**4) * tau2 * fi1
        fi1 *= -(tau2**4) * tau1

        fi2 = 1.0 / ((tau1**2 - tau2**2) ** 3)
        fj2 = -(tau1**6 - 3 * tau1**4 * tau2**2) * fi2
        fi2 *= tau2**6 - 3 * tau2**4 * tau1**2

        expi = exp(-tau1 * dist)
        expj = exp(-tau2 * dist)

        value = expi * (fi1 + fi2 / dist) + expj * (fj1 + fj2 / dist)
        derivative = (
            np.atleast_1d(
                expi * (tau1 * (fi1 + fi2 / dist) + fi2 / (dist**2))
                + expj * (tau2 * (fj1 + fj2 / dist) + fj2 / (dist**2))
            )[:, None]
            * rho
        )
    return value, np.squeeze(derivative)


corrections = {"gauss": gauss_correction, "slater": slater_correction}


class CoulombTable(SetupConsistent):
    def __init__(self, setups: Setups, correction: str = "gauss") -> None:
        super().__init__(setups)

        # arrays
        atype = np.ndarray
        self.metarrays = {
            "g": (atype, dict(shape=("natoms", "natoms"), dtype=float)),
            "dg": (atype, dict(shape=("natoms", "natoms", 3), dtype=float)),
        }
        self.corrtype = correction

    gamma = arrayproperty("g", "Gamma")
    dgamma = arrayproperty("dg", "Gamma derivative")

    def calculate(self):
        """Construct the gamma matrix and its derivative."""
        # TODO
        # - add gamma cutoff
        correction = corrections[self.corrtype]
        for el1 in self.setups:
            self.gamma[el1.index, el1.index] = el1.U
            for el2, rho, dist in el1.neighbors:
                g, dg = correction(el1.U, el2.U, dist, rho)
                self.gamma[el1.index, el2.index] = g
                self.gamma[el2.index, el1.index] = g
                self.dgamma[el1.index, el2.index] = dg
                self.dgamma[el2.index, el1.index] = -dg


class Electrostatic(SimpleCalculator):
    def __init__(self, setups: Setups, correction: str = "gauss") -> None:
        super().__init__(setups)
        self.coulomb = CoulombTable(setups, correction)

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
            vpot[el1.index] = gamma[el1.index, :].dot(dq[:])  # (-1) for correct sign
            F[el1.index] = -dq[el1.index, None] * dq[:].dot(dgamma[el1.index, :])

        # energy
        self.E = 0.5 * dq.dot(vpot)

        # hamiltonian
        H = self.H
        for el1 in self.setups:
            o1 = el1.orbitals_slice
            # diagonal
            for e in range(o1.start, o1.stop):
                H[e, e] = vpot[el1.index]  # off-diag vanish when mult. by S
            # off-diagonal
            # TODO: improve read locality (scan row-wise)
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
