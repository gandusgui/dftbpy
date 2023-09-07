from itertools import permutations
from math import acos, cos, sin, sqrt, tan

import numpy as np
from ase import Atom

slako_integrals = ["dds", "ddp", "ddd", "pds", "pdp", "pps", "ppp", "sds", "sps", "sss"]


def select_integrals(nlf1_j, nlf2_j):
    selected = []
    for ski, (l1, l2, itype) in enumerate(slako_integrals):
        s1_j = []
        for nlf in nlf1_j:
            if l1 == nlf[1]:
                s1_j.append(nlf)
        if len(s1_j) == 0:
            continue
        s2_j = []
        for nlf in nlf2_j:
            if l2 == nlf[1]:
                s2_j.append(nlf)
        if len(s2_j) == 0:
            continue
        for nl1 in s1_j:
            for nl2 in s2_j:
                selected.append((ski, nl1, nl2))
    return selected


def g(t1, t2):
    """Angle-dependent part of the two-center integral."""
    c1, c2, s1, s2 = np.cos(t1), np.cos(t2), np.sin(t1), np.sin(t2)
    return [
        5.0 / 8 * (3 * c1**2 - 1) * (3 * c2**2 - 1),
        15.0 / 4 * s1 * c1 * s2 * c2,
        15.0 / 16 * s1**2 * s2**2,
        sqrt(15.0) / 4 * c1 * (3 * c2**2 - 1),
        sqrt(45.0) / 4 * s1 * s2 * c2,
        3.0 / 2 * c1 * c2,
        3.0 / 4 * s1 * s2,
        sqrt(5.0) / 4 * (3 * c2**2 - 1),
        sqrt(3.0) / 2 * c2,
        0.5,
    ]


def make_grid(Rz, nt, nr, rcut, rmin=1e-7, p=2, q=2):
    """make polar grid for two-center integrals."""
    h = Rz / 2
    T = np.linspace(0, 1, nt) ** p * np.pi
    R = rmin + np.linspace(0, 1, nr) ** q * (rcut - rmin)

    grid = []
    area = []
    # first calculate grid for polar centered on atom 1:
    # the z=h-like starts cutting full elements starting from point (1)
    for j in range(nt - 1):
        for i in range(nr - 1):
            # corners of area element
            z1 = R[i + 1] * cos(T[j])
            z2 = R[i] * cos(T[j])
            z3 = R[i] * cos(T[j + 1])
            z4 = R[i + 1] * cos(T[j + 1])
            A0 = (R[i + 1] ** 2 - R[i] ** 2) * (T[j + 1] - T[j]) / 2

            if z1 <= h:
                # area fully inside region
                r0 = 0.5 * (R[i] + R[i + 1])
                t0 = 0.5 * (T[j] + T[j + 1])
                A = A0
            elif z1 > h and z2 <= h and z4 <= h:
                # corner 1 outside region
                Th = acos(h / R[i + 1])
                r0 = 0.5 * (R[i] + R[i + 1])
                t0 = 0.5 * (Th + T[j + 1])
                A = A0
                A -= 0.5 * R[i + 1] ** 2 * (Th - T[j]) - 0.5 * h**2 * (
                    tan(Th) - tan(T[j])
                )
            elif z1 > h and z2 > h and z3 <= h and z4 <= h:
                # corners 1 and 2 outside region
                Th1 = acos(h / R[i])
                Th2 = acos(h / R[i + 1])
                r0 = 0.5 * (R[i] + R[i + 1])
                t0 = 0.5 * (Th2 + T[j + 1])
                A = A0
                A -= A0 * (Th1 - T[j]) / (T[j + 1] - T[j])
                A -= 0.5 * R[i + 1] ** 2 * (Th2 - Th1) - 0.5 * h**2 * (
                    tan(Th2) - tan(Th1)
                )
            elif z1 > h and z2 > h and z4 > h and z3 <= h:
                # only corner 3 inside region
                Th = acos(h / R[i])
                r0 = 0.5 * (R[i] + h / cos(T[j + 1]))
                t0 = 0.5 * (Th + T[j + 1])
                A = 0.5 * h**2 * (tan(T[j + 1]) - tan(Th)) - 0.5 * R[i] ** 2 * (
                    T[j + 1] - Th
                )
            elif z1 > h and z4 > h and z2 <= h and z3 <= h:
                # corners 1 and 4 outside region
                r0 = 0.5 * (R[i] + h / cos(T[j + 1]))
                t0 = 0.5 * (T[j] + T[j + 1])
                A = 0.5 * h**2 * (tan(T[j + 1]) - tan(T[j])) - 0.5 * R[i] ** 2 * (
                    T[j + 1] - T[j]
                )
            elif z3 > h:
                A = -1
            else:
                raise RuntimeError("Illegal coordinates.")
            d, z = (r0 * sin(t0), r0 * cos(t0))
            if (
                A > 0
                and sqrt(d**2 + z**2) < rcut
                and sqrt(d**2 + (Rz - z) ** 2) < rcut
            ):
                grid.append([d, z])
                area.append(A)

    # calculate the polar centered on atom 2 by mirroring the other grid
    grid = np.array(grid)
    area = np.array(area)
    grid2 = grid.copy()
    grid2[:, 1] = -grid[:, 1]
    shift = np.zeros_like(grid)
    shift[:, 1] = 2 * h
    grid = np.concatenate((grid, grid2 + shift))
    area = np.concatenate((area, area))

    return grid.T.copy(), area


class SlaterKosterTable:
    """Slater-Koster table for two atoms."""

    def __init__(self, atom1: Atom, atom2: Atom = None):
        atom2 = atom1 if atom2 is None else atom2
        s1 = atom1.symbol
        s2 = atom2.symbol
        self.atoms = {s1: atom1, s2: atom2}
        self.pairs = set(permutations([s1, s2]))

        funcs = {}
        for s in (s1, s2):
            atom = self.atoms[s]
            funcs[s] = {
                "v": atom.spline(atom.v),
                "R_j": [atom.spline(R) for R in atom.R_j],
                "K_j": [
                    atom.spline(atom.rgd.kin2(R, l)) for R, l in zip(atom.R_j, atom.l_j)
                ],
            }
        self.funcs = funcs

    def run(self, N=50, nt=150, nr=50):
        rcut = max(atom.get_cutoff() for atom in self.atoms.values())
        rmin = 1e-7
        # Atomic distances
        R1 = 0.05
        R2 = sum(atom.get_cutoff() for atom in self.atoms.values())
        self.R = np.linspace(R1, R2, N, endpoint=True)
        # Slater-Koster tables
        self.tables = {(s1, s2): np.zeros((N, 20)) for s1, s2 in self.pairs}

        for iR, Rz in enumerate(self.R):
            (d, z), dA = make_grid(Rz, nt, nr, rcut=rcut, rmin=rmin)
            ddA = dA * d  # polar integration factor

            # precomputations
            r1 = np.sqrt(d**2 + z**2)
            r2 = np.sqrt(d**2 + (Rz - z) ** 2)
            # Angular integration
            P = g(np.arccos(z / r1), np.arccos((z - Rz) / r2))

            for s1, s2 in self.pairs:
                V = (
                    self.funcs[s1]["v"](r1)
                    - self.atoms[s1].vconf(r1)
                    + self.funcs[s2]["v"](r2)
                    - self.atoms[s2].vconf(r2)
                )  # should be approx. to the true crystal potential
                skt = self.tables[(s1, s2)]
                R1_j = self.funcs[s1]["R_j"]
                R2_j = self.funcs[s2]["R_j"]
                K2_j = self.funcs[s2]["K_j"]

                selected = select_integrals(
                    self.atoms[s1].valence_configuration,
                    self.atoms[s2].valence_configuration,
                )

                skt = self.tables[(s1, s2)]

                for ski, nlf1, nlf2 in selected:
                    j1 = self.atoms[s1].index(nlf1)
                    j2 = self.atoms[s2].index(nlf2)
                    R1 = R1_j[j1](r1)
                    R2 = R2_j[j2](r2)
                    K2 = K2_j[j2](r2)

                    S = np.dot(R1 * R2 * P[ski], ddA)
                    H = np.dot(R1 * (K2 + V * R2) * P[ski], ddA)

                    skt[iR, ski] = H
                    skt[iR, ski + 10] = S

    def __getitem__(self, key):
        return self.tables[key]
