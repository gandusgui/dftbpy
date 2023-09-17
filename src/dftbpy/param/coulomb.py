import pprint
from pathlib import Path

import numpy as np

from dftbpy.param.atom import Atom
from dftbpy.param.hartree import hartree
from dftbpy.param.slako import make_grid


class Hubbard:
    def __init__(self, atom: Atom, **kwargs):
        if isinstance(atom, str):
            atom = Atom(atom, **kwargs)
        self.atom = atom
        self.nlf_j = tuple(atom.valence_configuration.keys())
        self.nj = len(self.nlf_j)
        self.arrays = {
            "dn": np.zeros((self.nj, atom.N)),  # Density change
            "U": np.zeros((self.nj, self.nj)),  # Coulomb tensor
            "dv": np.zeros((self.nj, atom.N)),  # Hartree change
        }
        self.U = self.arrays["U"]
        self.dn = {}
        self.dv = {}
        self.labels = np.empty((self.nj, self.nj), dtype=object)
        for i, j in np.ndindex((self.nj, self.nj)):
            pair = (self.nlf_j[i], self.nlf_j[j])
            self.labels[i, j] = pair
            if i == j:
                self.dn[self.nlf_j[i]] = self.arrays["dn"][i]
                self.dv[self.nlf_j[i]] = self.arrays["dv"][i]

        self.nupdates = 0

    def initialize(self):
        atom = self.atom
        if atom.nupdates == 0:
            atom.run()
        self.e_j = atom.e_j.copy()
        self.f_j = atom.f_j.copy()
        self.n = atom.calculate_density()
        # self.n_j = atom.R_j**2 / (4 * np.pi)  #

    def integrate(self, x):
        r = self.atom.rgd.r_g
        return (4 * np.pi) * np.trapz(x * r**2, r)

    def compute(self, nlf1, nlf2=None, df=0.01):
        # de[i] / df[j], dn[i] / df[j]
        nlf2 = nlf2 or nlf1
        atom = self.atom
        j1 = atom.index(nlf1)
        j2 = atom.index(nlf2)
        e1 = self.e_j[j1]
        # fraction of an electron is removed
        # for partially occupied orbitals and
        # is added for unoccupied orbitals.
        f2 = self.f_j[j2]
        virtual = round(f2, 3) == 0
        occupied = not virtual
        sign = -1 if occupied else +1  # if virtual
        fp2 = f2 + sign * df
        atom.f_j[j2] = fp2
        atom.run()
        ep1 = atom.e_j[j1]
        de1df2 = (ep1 - e1) / (fp2 - f2)
        dn1df2 = (atom.calculate_density() - self.n) / (fp2 - f2)
        # dn1df2 = (atom.R_j[j1] ** 2 / (4 * np.pi) - self.n_j[j1]) / (fp2 - f2)
        # reset
        atom.f_j[j2] = f2
        atom.e_j[:] = self.e_j
        return de1df2, dn1df2

    def run(self, df=0.01):
        self.initialize()
        r = self.atom.rgd.r_g
        dr = self.atom.rgd.dr_g
        r1 = r[1]
        r2 = r[2]
        for i, j in zip(*np.triu_indices(self.nj)):
            nlf1, nlf2 = self.labels[i, j]
            dedf, dndf = self.compute(nlf1, nlf2, df=df)
            de2df, dn2df = self.compute(nlf1, nlf2, df=2 * df)
            # linear interpolation to df -> 0
            self.U[i, j] = dedf - (de2df - dedf)
            if i == j:
                self.dn[nlf1] = dn = dndf - (dn2df - dndf)
                dv = self.dv[nlf1]
                hartree(0, dn * r * dr, r, dv)
                dv[1:] /= r[1:]
                # mid-point formula
                dv[0] = 0.5 * (dv[1] + dv[2] + (dv[1] - dv[2]) * (r1 + r2) / (r2 - r1))
            self.U[j, i] = self.U[i, j]

        self.nupdates += 1

    def get_cutoff(self):
        gcut = max(self.atom.rgd.get_cutoff(dv) for dv in self.dv.values())
        return self.atom.rgd.r_g[gcut - 1]

    def spline(self, x):
        return self.atom.spline(x)

    def todict(self):
        d = {}
        d.update(self.atom.todict())
        d["U"] = self.U
        return d

    def __repr__(self) -> str:
        return pprint.PrettyPrinter(compact=True).pformat(self.todict())

    def write(self, dir="."):
        path = Path(dir)
        name = self.symbol
        if path.suffix:
            # make sure .elm
            name = path.name
            path = path.parent
        file = (path / name).with_suffix(".elm")
        with open(file, "w") as fp:
            pprint.PrettyPrinter(compact=True, stream=fp).pprint(self.todict())


class CoulombTable:
    """Slater-Koster table for two atoms."""

    def __init__(self, hub1: Hubbard, hub2: Hubbard = None):
        hub2 = hub1 if hub2 is None else hub2
        s1 = hub1.atom.symbol
        s2 = hub2.atom.symbol
        self.hubs = {s1: hub1, s2: hub2}
        self.pair = [s1, s2]

        self.dn = dn = {}
        self.dv = dv = {}
        for s, hub in self.hubs.items():
            dn[s] = {nlf: hub.spline(dn) for nlf, dn in hub.dn.items()}
            dv[s] = {nlf: hub.spline(dv) for nlf, dv in hub.dv.items()}

        n1 = len(self.dn[s1])
        n2 = len(self.dn[s2])
        self.labels = np.empty((n1, n2), dtype=object)

        for i, nlf1 in enumerate(self.dn[s1]):
            for j, nlf2 in enumerate(self.dn[s2]):
                self.labels[i, j] = (nlf1, nlf2)

    @property
    def shape(self):
        return self.labels.shape

    def initialize(self):
        for hub in self.hubs.values():
            if hub.nupdates == 0:
                hub.run()

    def run(self, R1=None, R2=None, N=50, nt=150, nr=50):
        self.initialize()
        coffs = [hub.get_cutoff() for hub in self.hubs.values()]
        rmin = 1e-7
        R1 = R1 or 0.05
        # min(
        #     hub.atom.rgd.r_g[find_first_peakpos(dn)]
        #     for hub in self.hubs.values()
        #     for dn in hub.dn.values()
        # )
        R2 = R2 or sum(coffs)
        rcut = R2 * 1.2
        self.R = np.linspace(R1, R2, N, endpoint=True)
        # Coulomb tables
        self.tables = np.zeros((N,) + self.shape, dtype=float)

        s1, s2 = self.pair

        for iR, Rz in enumerate(self.R):
            (d, z), dA = make_grid(Rz, nt, nr, rcut=rcut, rmin=rmin)
            ddA = dA * d  # polar integration factor

            # precomputations
            r1 = np.sqrt(d**2 + z**2)
            r2 = np.sqrt(d**2 + (Rz - z) ** 2)

            table = self.tables[iR]

            for i, j in np.ndindex(self.shape):
                nlf1, nlf2 = self.labels[i, j]
                dn1 = self.dn[s1][nlf1](r1)
                dv2 = self.dv[s2][nlf2](r2)

                table[i, j] = np.dot(dn1 * dv2, ddA)

    def __getitem__(self, index):
        return self.table[index]

    def todict(self):
        d = {}
        d["R"] = self.R
        for symbol in self.pair:
            d[symbol] = self.hubs[symbol]
        d["tables"] = self.tables
        return d

    def __repr__(self) -> str:
        return pprint.PrettyPrinter(compact=True).pformat(self.todict())

    def write(self, dir="."):
        path = Path(dir)
        name = next("-".join(pair) for pair in self.pairs)
        if path.suffix:
            # make sure .elm
            name = path.name
            path = path.parent
        file = (path / name).with_suffix(".slako")
        with open(file, "w") as fp:
            pprint.PrettyPrinter(compact=True, stream=fp).pprint(self.todict())


def find_first_peakpos(f):
    df = f[1:] - f[:-1]  # derivative
    return np.argwhere((df[1:] * df[:-1]) < 0)[0][0]  # 1st change of sign (peak)
