from typing import Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.io import read
from ase.units import Bohr

from dftbpy.setups import NeighborList

# WORK IN PROGRESS
# from scipy.spatial.distance import pdist, squareform
# dist = squareform(pdist(atoms.positions))
# d = np.unique(dist.round(round))
# np.digitize(dist, d)


def find_pair_neighbors(atoms, pair, rcut):
    # find pair closer than cutoff
    indices = atoms.symbols.search(pair)
    cutoffs = {pair: rcut}
    if pair[0] != pair[1]:
        cutoffs[pair[0], pair[0]] = np.inf
        cutoffs[pair[1], pair[1]] = np.inf
    nl = NeighborList(cutoffs)
    nl.build(
        atoms.pbc, atoms.cell, atoms.positions[indices] / Bohr, atoms.numbers[indices]
    )
    return nl.i, nl.j, nl.d


def find_constraint_subsets(atoms, pair, rcut, rtol=3):
    # split neighbor pairs in subsets with equal `round` distance.
    i, j, d = find_pair_neighbors(atoms, pair, rcut)
    d, inv, count = np.unique(d.round(rtol), return_counts=True, return_inverse=True)
    constraints = []
    for ix in range(d.size):
        fix = inv != ix  # all other pair neighs
        c = FixBondLengths(tuple(zip(i[fix], j[fix])) if any(fix) else [])
        constraints.append(c)
    return d, count, constraints


class RepulsiveFitting(list):
    def __init__(
        self, pair: Tuple[str, str], rcut: float, s: float = None, k: int = 3, rtol=3
    ) -> None:
        self.pair = tuple(pair)  # symbols
        self.rcut = rcut / Bohr  # [Ang] cutoff V_rep(R>Rep)=0
        self.s = s  # smoothing par
        self.k = k  # order of spline for V'_rep(R)
        self.scale = 1.025  # scaling factor for equilibrium systems
        self.rtol = rtol  # Rounding tolerance

    def append(self, d, dvrep, weight=1.0):
        super().append((d, dvrep, weight))

    def append_scalable_system(self, atoms: Atoms, calc, weight=1.0):
        if isinstance(atoms, str):
            atoms = read(atoms)
        assert atoms.cell.any(), "Cell unset."
        dist, count, constr = find_constraint_subsets(
            atoms, self.pair, self.rcut, self.rtol
        )
        e1 = calc.get_potential_energy(atoms)
        old = atoms.copy()
        for d, N, c in zip(dist, count, constr):
            atoms = old.copy()
            atoms.set_cell(atoms.get_cell() * self.scale, scale_atoms=True)
            c.adjust_positions(old, atoms.positions)
            e2 = calc.get_potential_energy(atoms)
            dvrep = -(e2 - e1) / (d * self.scale - d) / N
            self.append(d, dvrep, weight)

    def append_dimer(self, d, calc, weight=1.0):
        atoms = Atoms(self.pair, positions=[[0, 0, 0], [d, 0, 0]])
        atoms.center(vacuum=5.0)
        self.append_scalable_system(atoms, calc, weight)
