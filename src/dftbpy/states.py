from warnings import warn

import numpy as np
from scipy.linalg import eigh

from dftbpy.electrostatic import Electrostatic
from dftbpy.occupations import FermiDirac
from dftbpy.potential import Potential


class States:
    def __init__(
        self, potential: Potential, electrostatic: Electrostatic = None, charge: int = 0
    ) -> None:
        self.potential = potential
        self.electrostatic = electrostatic
        self.setups = potential.setups
        self.distribution = FermiDirac(width=0.0)
        self.charge = charge

        self.F = np.zeros((len(self.setups), 3))

    @property
    def scc(self):
        return self.electrostatic is not None

    def solve(self, dq=None):
        """Solve eigenstates."""
        pot = self.potential
        elecs = self.electrostatic
        dist = self.distribution

        # construct Hamiltonian
        H0 = pot.H
        S = pot.S
        if self.scc:
            elecs.update(dq)  # , system_changes=["charges"])
            H = H0 + elecs.H * S
        else:
            H = H0

        # solve states
        #  TODO: check that phis.T @ S @ phis = I
        eigs, wfs = eigh(H, S)
        # occupy states
        fermi_level = dist.calculate_fermi_level(eigs, self.setups.nel - self.charge)
        f, dfde = dist.occupy(eigs, fermi_level)
        # denisty matrix
        rho = np.einsum("i,ji,ki->jk", f, wfs, wfs, optimize=True)
        rhoe = np.einsum(
            "i,i,ji,ki->jk", eigs, f, wfs, wfs, optimize=True
        )  # needed later
        # mulliken charges
        self.q = q = np.einsum("ij,ji->i", rho, S, optimize=True)
        for el1 in self.setups:
            dq[el1.index] = q[el1.orbitals_slice].sum() - el1.nel

        self.f = f
        self.eigs = eigs
        self.wfs = wfs
        self.fermi_level = fermi_level
        self.rho = rho
        self.rhoe = rhoe

        return dq

    def update(self):
        """Calculate states, energy and forces"""
        pot = self.potential
        elecs = self.electrostatic

        if self.scc:
            # solve self-consistent
            dq_inp = np.zeros(len(self.setups))  # initial guess
            self.niter = 0
            while True:
                dq_out = self.solve(dq_inp.copy())
                self.eps = abs(dq_out - dq_inp).mean()
                if self.eps < 1e-5 or self.niter > 50:
                    break
                dq_inp = dq_out  # XXX could be linear mixing
                self.niter += 1

            self.dq = dq_out
        else:
            self.dq = self.solve()

        # decrease/increase of electrons must compensate system's charge
        charge = -self.dq.sum()
        if abs(charge - self.charge) > 1e-3:
            warn(
                f"Change in electrons is {-charge:.3f} and does not "
                f"correspond to the system's charge state {self.charge:.3f}.",
                UserWarning,
            )

        # Forces
        if self.scc:
            dH = pot.dH + elecs.H[..., None] * pot.dS
        else:
            dH = pot.dH
        orbforces = -np.einsum("ij,jik->ik", self.rho, dH, optimize=True) + np.einsum(
            "ij,jik->ik", self.rhoe, pot.dS, optimize=True
        )
        for el1 in self.setups:
            self.F[el1.index] = orbforces[el1.orbitals_slice].sum(0)
        # Energy
        self.E = np.einsum("ij,ji", self.rho, pot.H, optimize=True)  # Trace[rho H]
        if self.scc:
            self.E += elecs.E
            self.F += elecs.F
