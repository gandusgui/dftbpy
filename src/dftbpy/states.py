from warnings import warn

import numpy as np
from scipy.linalg import eigh

from dftbpy.calculator import SetupConsistent, all_properties, arrayproperty
from dftbpy.electrostatic import Electrostatic
from dftbpy.occupations import FermiDirac
from dftbpy.potential import Potential


class States(SetupConsistent):
    implemented_properties = all_properties

    def __init__(
        self,
        setups,
        scc: bool = True,
        charge: int = 0,
        width: float = 0.0,
        correction: str = "gauss",
    ) -> None:
        super().__init__(setups)

        self.pot = Potential(setups)
        self.elecs = Electrostatic(setups, correction)

        self.dist = FermiDirac(width=width)
        self.charge = charge
        self.scc = scc

        # arrays
        atype = np.ndarray
        self.metarrays = {
            "P": (atype, dict(shape=("no", "no"), dtype=float)),
            "F": (atype, dict(shape=("natoms", 3), dtype=float)),
            "dq": (atype, dict(shape=("natoms",), dtype=float)),
        }

    P = arrayproperty("P", "Density Matrix")
    F = arrayproperty("F", "Forces")
    dq = arrayproperty("dq", "Mulliken charges")

    def solve_eigenstates(self, dq=None):
        """Compute eigenstates, density matrix and mulliken charges."""
        pot = self.pot
        elecs = self.elecs
        dist = self.dist

        # construct Hamiltonian
        pot.update()
        H0 = pot.H
        S = pot.S
        if self.scc:
            elecs.update(dq)
            H = H0 + elecs.H * S
        else:
            H = H0

        # solve states
        #  TODO: check that phis.T @ S @ phis = I
        e, wfs = eigh(H, S)
        # occupy states
        fermi_level = dist.calculate_fermi_level(e, self.setups.nel - self.charge)
        f, dfde = dist.occupy(e, fermi_level)
        # denisty matrix
        P = np.einsum("i,ji,ki->jk", f, wfs, wfs, out=self.P, optimize=True)
        # mulliken charges
        q = np.einsum("ij,ji->i", P, S, optimize=True)
        dq = self.dq
        for el1 in self.setups:
            # change in number of electrons
            dq[el1.index] = q[el1.orbitals_slice].sum() - el1.nel

        self.f = f
        self.e = e
        self.wfs = wfs
        self.fermi_level = fermi_level

        return dq

    def solve(self):
        """Solve (self-consistent) charge"""
        if self.scc:
            # self-consistent
            dq_inp = np.zeros(len(self.setups))  # initial guess
            self.niter = 0
            while True:
                dq_out = self.solve_eigenstates(dq_inp)
                self.eps = abs(dq_out - dq_inp).mean()
                if self.eps < 1e-5 or self.niter > 50:
                    break
                dq_inp = dq_out  # XXX could be linear mixing
                self.niter += 1

            dq = dq_out
        else:  # non self-consistent
            dq = self.solve_eigenstates()

        # consistent electron change
        charge = -dq.sum()
        if abs(charge - self.charge) > 1e-3:
            warn(
                f"Change in electrons is {-charge:.3f} and does not "
                f"correspond to the system's charge state {self.charge:.3f}.",
                UserWarning,
            )

    def calculate(self):
        """Calculate states, energy and forces"""
        pot = self.pot
        elecs = self.elecs

        # solve (self-consistent)
        self.solve()

        # forces
        F = self.F
        if self.scc:
            dH = pot.dH + elecs.H[..., None] * pot.dS
        else:
            dH = pot.dH
        Pe = np.einsum(
            "i,i,ji,ki->jk", self.e, self.f, self.wfs, self.wfs, optimize=True
        )
        orbF = -np.einsum("ij,jik->ik", self.P, dH, optimize=True) + np.einsum(
            "ij,jik->ik", Pe, pot.dS, optimize=True
        )  # orbital forces Trace[-rho dH + e rho dS]
        for el1 in self.setups:
            F[el1.index] = orbF[el1.orbitals_slice].sum(0)
        if self.scc:  # + 1/2 sum_IJ dgamma_IJ dQ_J
            F += elecs.F

        # energy
        self.E = np.einsum("ij,ji", self.P, pot.H, optimize=True)  # Trace[rho H]
        if self.scc:
            self.E += elecs.E
        self.E -= self.setups.E
