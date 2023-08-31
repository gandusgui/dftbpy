import numpy as np
from ase.atoms import Atoms
from ase.neighborlist import first_neighbors, primitive_neighbor_list
from ase.units import Bohr

from dftbpy.param import SlaterKosterTable
from dftbpy.param.atom import Atom
from dftbpy.param.configs import angular_number
from dftbpy.slako import SlaterKosterParam


class NeighborList:
    def __init__(self, cutoffs, skin=0.3):
        self.cutoffs = cutoffs
        self.bothways = False
        self.skin = skin  # Any change in positions dp | dp < skin^2 triggers update
        self.self_interaction = False
        self.sorted = True
        self.nupdates = 0

    def update(self, atoms: Atoms):
        """Make sure the list is up to date."""
        pbc = atoms.pbc
        cell = atoms.cell
        positions = atoms.positions / Bohr  # ase.Atoms conventions
        numbers = atoms.numbers

        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers)
            return True

        if (
            (self.pbc != pbc).any()
            or (self.cell != cell).any()
            or ((self.positions - positions) ** 2).sum(1).max() > self.skin**2
        ):
            self.build(pbc, cell, positions, numbers)
            return True

        return False

    def build(self, pbc, cell, positions, numbers):
        """Build the list."""
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)
        i, j, d, D, S = primitive_neighbor_list(
            "ijdDS",
            pbc,
            cell,
            positions,
            cutoff=self.cutoffs,
            numbers=numbers,
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
        self.d = d
        self.rho = D / d[:, None]  # self_interaction=False
        self.D = D

        self.first_neigh = first_neighbors(len(positions), i)

        self.nupdates += 1

    def get_neighbors(self, a):
        """Return the neighbors of atom a.

        Returns:
            j: indices of neighbors
            rho: distance unit vector to neighbors,
                i.e. abs(rho)'s = 1.
            d: absolute distance to neighbors,
                i.e. positions[a] + d * rho = positions of a' neighbors.
        """
        return (
            self.j[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.rho[self.first_neigh[a] : self.first_neigh[a + 1]],
            self.d[self.first_neigh[a] : self.first_neigh[a + 1]],
        )


class Setups:
    """Setups for a system of atoms."""

    def __init__(self, slakos: dict[SlaterKosterTable]) -> None:
        self.slakos = {}  # Slater-Koster pairs
        self.elements = {}  # Unique elements
        self.cutoffs = {}  # Cutoff distances pairs
        for (s1, s2), skt in slakos.items():  # skt = SlaterKosterTable
            self.slakos[s1, s2] = slako12 = SlaterKosterParam(skt, (s1, s2))
            self.slakos[s2, s1] = slako21 = SlaterKosterParam(skt, (s2, s1))

            for symbol in (s1, s2):
                if symbol not in self.elements:
                    self.elements[symbol] = Setup(atom=skt.atoms[symbol])

            self.cutoffs[s1, s2] = slako12.cutoff
            self.cutoffs[s2, s1] = slako21.cutoff

        self.nl = NeighborList(cutoffs=self.cutoffs)
        self.symbols = np.ndarray(0, "U1")

    @property
    def nupdates(self):
        return self.nl.nupdates

    def update(self, atoms: Atoms):
        if self.nupdates == 0:
            self.symbols = np.array(atoms.symbols, copy=True)
            self.orbitals_slices = []
            self.nel = 0  # total number of electrons
            self.no = 0  # total number of orbitals
            for s in self.symbols:
                el = self.elements[s]
                self.orbitals_slices.append(slice(self.no, self.no + el.no))
                self.nel += el.nel
                self.no += el.no

        return self.nl.update(atoms)

    def get_total_number_of_orbitals(self):
        return self.no

    def get_total_number_of_electrons(self):
        return self.nel

    def get_neighbors(self, a):
        return self.nl.get_neighbors(a)

    @property
    def natoms(self):
        return len(self.symbols)

    def __len__(self):
        return self.natoms

    def __getitem__(self, a):
        return Setup(index=a, setups=self)

    def __iter__(self) -> iter:
        yield from (self[i] for i in range(len(self)))


def setupproperty(name, doc):
    """Helper function to easily create Setup attribute property."""

    def getter(self):
        return self.get(name)

    def setter(self, value):
        self.set(name, value)

    def deleter(self):
        self.delete(name)

    return property(getter, setter, deleter, doc)


class Setup:
    """Setup for an atom or element."""

    def __init__(
        self, /, atom: Atom = None, index: int = None, setups: Setups = None
    ) -> None:
        self.data = d = {}

        if setups is None:
            eners = []  # energies
            d["symbol"] = atom.symbol
            d["U"] = getattr(atom, "U", 0.3)  # backward compatibility
            no = 0
            nel = 0
            for nlf, e in atom.valence_configuration.items():
                no_ = 2 * angular_number[nlf[1]] + 1
                no += no_
                nel += int(nlf[2:])
                eners.extend(no_ * [e])
            d["no"] = no
            d["nel"] = nel
            d["energies"] = np.array(eners)

        self.index = index
        self.setups = setups

    def get(self, name):
        """Get name attribute."""
        if self.setups is None:
            return self.data[name]

        setups = self.setups
        symbol = setups.symbols[self.index]

        if name == "orbitals_slice":
            return setups.orbitals_slices[self.index]

        if name == "neighbors":
            return (
                (setups[nn], rho, dist)
                for nn, rho, dist in zip(*setups.get_neighbors(self.index))
            )

        return setups.elements[symbol].data[name]

    def set(self, name, value):
        """Set name attribute."""
        if self.setups is None:
            self.data[name] = value

    symbol = setupproperty("symbol", "Chemical symbol")
    no = setupproperty("no", "Number of orbitals")
    nel = setupproperty("nel", "Number of electrons")
    U = setupproperty("U", "Hubbard parameter")
    energies = setupproperty("energies", "Orbital energies")
    orbitals_slice = setupproperty("orbitals_slice", "Orbital indices")
    neighbors = setupproperty("neighbors", "Neighbors")
