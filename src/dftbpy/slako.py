from itertools import permutations
from math import acos, cos, sin, sqrt, tan

import numpy as np
from ase import Atom

angular_number = {"s": 0, "p": 1, "d": 2}
angular_name = {0: "s", 1: "p", 2: "d"}

slako_integrals = ["dds", "ddp", "ddd", "pds", "pdp", "pps", "ppp", "sds", "sps", "sss"]


def select_integrals(nl1_j, nl2_j):
    selected = []
    for integral in slako_integrals:
        l1 = angular_number[integral[0]]
        l2 = angular_number[integral[1]]
        s1_j = []
        for nl in nl1_j:
            if l1 == nl[1]:
                s1_j.append(nl)
        if len(s1_j) == 0:
            continue
        s2_j = []
        for nl in nl2_j:
            if l2 == nl[1]:
                s2_j.append(nl)
        if len(s2_j) == 0:
            continue
        for nl1 in s1_j:
            for nl2 in s2_j:
                selected.append((integral, nl1, nl2))
    return selected


def g(t1, t2):
    """Angle-dependent part of the two-center integral."""
    c1, c2, s1, s2 = cos(t1), cos(t2), sin(t1), sin(t2)
    return np.array(
        [
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
    )


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

    return grid, area


class SlaterKosterTable:
    def __init__(self, atom1: Atom, atom2: Atom):
        s1 = atom1.symbol
        s2 = atom2.symbol
        self.atoms = {s1: atom1, s2: atom2}
        self.pairs = set(permutations([s1, s2]))

    def run(self, R1, R2, N, nt=150, nr=50):
        rcut = max(atom.get_rcut() for atom in self.atoms.values())
        rmin = max(atom.get_rmin() for atom in self.atoms.values())
        # Atomic distances
        self.R = np.linspace(R1, R2, N, endpoint=True)
        # Slater-Koster tables
        self.skt = {(s1, s2): np.zeros((N, 20)) for s1, s2 in self.pairs}

        for iR, Rz in enumerate(self.R):
            grid, area = make_grid(Rz, nt, nr, rcut=rcut, rmin=rmin)

            # precomputations
            ng = len(grid)
            phi = np.empty((ng, 10))  # anglular integrals
            rho = np.empty((ng, 2))  # radii
            s1, s2 = self.atoms.keys()
            atom1 = self.atoms[s1]
            atom2 = self.atoms[s2]
            v1 = np.empty(ng)
            v2 = np.empty(ng)
            for ig, (d, z) in enumerate(grid):
                r1, r2 = sqrt(d**2 + z**2), sqrt(d**2 + (Rz - z) ** 2)
                t1, t2 = acos(z / r1), acos((z - Rz) / r2)
                phi[ig, :] = g(t1, t2)
                rho[ig, :] = r1, r2
                v1[ig] = atom1.potential(r1)  # - atom1.confinement(r1)
                v2[ig] = atom2.potential(r2)  # - atom2.confinement(r2)
            v = {s1: v1, s2: v2}

            for s1, s2 in self.pairs:
                skt = self.skt[(s1, s2)]
                atom1 = self.atoms[s1]
                atom2 = self.atoms[s2]
                v1 = v[s1]
                v2 = v[s1]
                selected = select_integrals(
                    atom1.get_valence_states(), atom2.get_valence_states()
                )

                # internal
                skH = np.zeros(10)
                skS = np.zeros(10)

                for slako, nl1, nl2 in selected:
                    ski = slako_integrals.index(slako)
                    state1 = atom1.states[nl1]
                    state2 = atom2.states[nl2]
                    l2 = nl2[1]

                    S = 0.0
                    H = 0.0

                    for ig, dA in enumerate(area):
                        r1 = rho[ig, 0]
                        r2 = rho[ig, 1]
                        R1 = state1.R(r1)
                        R2 = state2.R(r2)
                        d2udr2 = state2.u(r2, 2)

                        aux = phi[ig, ski] * dA * grid[ig, 0]
                        S += R1 * R2 * aux
                        H += (
                            R1
                            * (
                                -0.5 * d2udr2 / r2
                                + (v1[ig] + v2[ig] + l2 * (l2 + 1) / (2 * r2**2)) * R2
                            )
                            * aux
                        )

                    skH[ski] = H
                    skS[ski] = S

                skt[iR, :10] = skH[:]
                skt[iR, 10:] = skS[:]
