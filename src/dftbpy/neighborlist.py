import numpy as np
from ase.neighborlist import first_neighbors, primitive_neighbor_list


class NeighborList:
    def __init__(self, cutoffs, skin=0.3):
        self.cutoffs = cutoffs
        self.bothways = False
        self.skin = skin  # Any change in positions dp | dp < skin^2 triggers update
        self.self_interaction = False
        self.sorted = False
        self.nupdates = 0
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, atoms):
        """Make sure the list is up to date."""
        pbc = atoms.pbc
        cell = atoms.cell
        positions = atoms.positions

        if self.nupdates == 0:
            self.build(pbc, cell, positions)
            return True

        if (
            (self.pbc != pbc).any()
            or (self.cell != cell).any()
            or ((self.positions - positions) ** 2).sum(1).max() > self.skin**2
        ):
            self.build(pbc, cell, positions)
            return True

        return False

    def build(self, pbc, cell, positions):
        """Build the list."""
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)
        i, j, d, D, S = primitive_neighbor_list(
            "ijdDS",
            pbc,
            cell,
            positions,
            self.cutoffs,
            self_interaction=self.self_interaction,
            use_scaled_positions=False,
        )

        if len(positions) > 0 and not self.bothways:
            S_x, S_y, S_z = S.T

            mask = S_z > 0
            mask &= S_y == 0
            mask |= S_y > 0
            mask &= S_x == 0
            mask |= S_x > 0
            mask |= (i <= j) & (S == 0).all(axis=1)

            i = i[mask]
            j = j[mask]
            d = d[mask]
            D = D[mask]
            # S = S[mask]

        if len(positions) > 0 and self.sorted:
            mask = np.argsort(i * len(i) + j)
            i = i[mask]
            j = j[mask]
            d = d[mask]
            D = D[mask]
            # S = S[mask]

        self.i = i
        self.j = j
        self.rho = d / D  # self_interaction=False, no D==0.
        self.D = D

        self.first_neigh = first_neighbors(len(positions), i)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return the neighbors of atom a."""
        return (
            self.j[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.rho[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.D[self.first_neigh[a] : self.first_neigh[a + 1]],
        )
