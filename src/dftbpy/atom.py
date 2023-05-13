from collections import namedtuple
from math import log, pi, sqrt

import numpy as np
from ase.data import chemical_symbols

from dftbpy.configs import configurations, valence_states
from dftbpy.hartree import hartree
from dftbpy.spline import Spline
from dftbpy.xcf import LDA


class Grid:
    """r=ag/(1-bg), g=0,..,N-1"""

    def __init__(self, a, b, N) -> None:
        self.a = a
        self.b = b
        self.g = np.arange(N)
        self.r_g = self.a * self.g / (1 - self.b * self.g)  # radial grid
        self.dr_g = (self.b * self.r_g + self.a) ** 2 / self.a  # dr/dg

    def integrate(self, a_xg, n=0):
        assert n >= -2
        return np.dot(a_xg[..., 1:], (self.r_g ** (2 + n) * self.dr_g)[1:]) * (4 * pi)

    def d2gdr2(self):
        return -2 * self.a * self.b / (self.b * self.r_g + self.a) ** 3


class Atom:
    def __init__(self, symbol, gpernode=150) -> None:
        self.symbol = symbol
        self.Z, nlfe_j = configurations[symbol]

        self.n_j = [n for n, l, f, e in nlfe_j]
        self.l_j = [l for n, l, f, e in nlfe_j]
        self.f_j = [f for n, l, f, e in nlfe_j]
        self.e_j = [e for n, l, f, e in nlfe_j]
        self.nel = self.Z

        # xc
        self.xc = LDA()

        # grid
        maxnodes = max([n - l - 1 for n, l in zip(self.n_j, self.l_j)])
        self.N = (maxnodes + 1) * gpernode
        self.beta = 0.4
        self.rgd = Grid(self.beta / self.N, 1.0 / self.N, self.N)

        # orbs
        self.nj = len(self.n_j)
        self.u_j = np.zeros((self.nj, self.N))  # radial wave functions
        self.vr = np.zeros(self.N)  # potential times radius
        self.n = np.zeros(self.N)  # electron density

    @property
    def nlfe_j(self):
        yield from zip(self.n_j, self.l_j, self.f_j, self.e_j)

    def add(self, n, l, df=+1, e=None):
        """Add (df=+1) or remove (df=-1) electron."""
        self.nel += df
        j = 0
        for n_, l_, f_, e_ in self.nlfe_j:
            if n_ == n and l_ == l:
                break
            j += 1
        if j < len(self.n_j):
            self.f_j[j] = self.f_j[j] + df
            if e is not None:
                self.e_j[j] = e
            else:
                # read e_ from neighbor symbol.
                neighbor = chemical_symbols[self.nel]
                for n_, l_, f_, e_ in configurations[neighbor][1]:
                    if n_ == n and l_ == l:
                        self.e_j[j] = e_
                        break
            return
        self.n_j.append(n)
        self.l_j.append(l)
        self.f_j.append(df)
        if e is None:
            self.e_j.append(-1.0 * self.Z**2 / n**2)

    def guess_radials(self):
        r = self.rgd.r_g
        dr = self.rgd.dr_g
        # Initialize with Slater function:
        for l, e, u in zip(self.l_j, self.e_j, self.u_j):
            if self.symbol in ["Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"]:
                a = sqrt(-4.0 * e)
            else:
                a = sqrt(-2.0 * e)

            u[:] = r ** (1 + l) * np.exp(-a * r)
            norm = np.dot(u**2, dr)
            u *= 1.0 / sqrt(norm)

    def calculate_density(self):
        """Calculate radial electron density."""
        # sum_nl |Rnl(r)|**2/(4*pi)
        n = np.dot(self.f_j, np.where(abs(self.u_j) < 1e-160, 0, self.u_j) ** 2) / (
            4 * pi
        )
        n[1:] /= self.rgd.r_g[1:] ** 2
        n[0] = n[1]
        return n

    def solve_radials(self):
        r = self.rgd.r_g
        dr = self.rgd.dr_g
        vr = self.vr

        c2 = -((r / dr) ** 2)
        c1 = -self.rgd.d2gdr2() * r**2

        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j, self.e_j, self.u_j)):
            nodes = n - l - 1  # analytically expected number of nodes
            delta = -0.2 * e
            nn, A = shoot(u, l, vr, e, r, dr, c1, c2)
            # adjust eigenenergy until u has the correct number of nodes
            while nn != nodes:
                diff = np.sign(nn - nodes)
                while diff == np.sign(nn - nodes):
                    e -= diff * delta
                    nn, A = shoot(u, l, vr, e, r, dr, c1, c2)
                delta /= 2

            # adjust eigenenergy until u is smooth at the turning point
            de = 1.0
            while abs(de) > 1e-9:
                norm = np.dot(np.where(abs(u) < 1e-160, 0, u) ** 2, dr)
                u *= 1.0 / sqrt(norm)
                de = 0.5 * A / norm
                x = abs(de / e)
                if x > 0.1:
                    de *= 0.1 / x
                e -= de
                assert e < 0.0
                nn, A = shoot(u, l, vr, e, r, dr, c1, c2)
            self.e_j[j] = e
            u *= 1.0 / sqrt(np.dot(np.where(abs(u) < 1e-160, 0, u) ** 2, dr))

    def run(self):
        # grid
        r = self.rgd.r_g
        dr = self.rgd.dr_g
        # orbs
        vr = self.vr
        vr[:] = 0.0  # initialize potential
        self.guess_radials()
        n = self.n
        n[:] = self.calculate_density()  # density

        # mix
        niter = 0
        nitermax = 117
        qOK = log(1e-10)
        mix = 0.4

        vrold = None
        vHr = np.zeros(self.N)

        while True:
            # harten potential
            hartree(0, n * r * dr, r, vHr)
            # nuclear potential
            vHr -= self.Z
            # exchange-correlation potential
            # print(n.shape, vr.shape, r.shape, self.xc.vxc.shape)
            self.xc.compute(n)
            vr[:] = vHr + self.xc.vxc * r
            # confinement
            # TODO: implement confinement
            # mix
            if niter > 0:
                vr[:] = (1.0 - mix) * vrold + mix * vr
            vrold = vr.copy()
            # solve radial equations and calculate new density
            self.solve_radials()
            dn = self.calculate_density() - n
            n += dn
            # estimate error from the square of the density change integrated
            q = log(np.sum((r * dn) ** 2))
            if q < qOK:
                break
            niter += 1
            if niter > nitermax:
                raise RuntimeError("Maximum number of iterations exceeded!")

        # Energy contributions
        Ekin = 0.0
        for f, e in zip(self.f_j, self.e_j):
            Ekin += f * e

        self.Ecoul = 2 * pi * np.dot(n * r * (vHr + self.Z), dr)
        self.Ekin = Ekin - 4 * pi * np.dot(n * vr * r, dr)
        self.Exc = self.rgd.integrate(self.xc.exc)
        self.Enucl = -4 * pi * np.dot(n * r * self.Z, dr)
        self.Etot = self.Exc + self.Ecoul + self.Ekin + self.Enucl

    def get_total_energy(self):
        return sum(self.e_j)

    def get_number_of_electrons(self):
        return self.rgd.integrate(self.n)


class States:
    State = namedtuple("state", ["e", "f", "R", "ravg"])

    def __init__(self, atom: Atom) -> None:
        self.atom = atom
        valence = valence_states[atom.symbol]

        states = {}
        r = atom.rgd.r_g
        R = np.empty_like(r)
        for j, (n, l) in enumerate(zip(atom.n_j, atom.l_j)):
            if (n, l) in valence:
                ravg = atom.rgd.integrate(atom.u_j[j], 1)
                R[1:] = atom.u_j[j][1:] / r[1:]
                R[0] = R[1]
                states[n, l] = States.State(
                    atom.e_j[j], atom.f_j[j], Spline(r, R.copy()), ravg
                )
        self.states = states

    def __getitem__(self, key):
        return self.states[key]

    def get_quantum_numbers(self):
        return self.states.keys()


def shoot(u, l, vr, e, r, dr, c1, c2, gmax=None):
    c0 = l * (l + 1) + 2 * r * (vr - e * r)
    if gmax is None and np.alltrue(c0 > 0):
        raise RuntimeError("Bad initial electron density guess!")
    c1 = c1
    # vectors needed for numeric integration of diff. equation
    fm = 0.5 * c1 - c2
    fp = 0.5 * c1 + c2
    f0 = c0 - 2 * c2

    if gmax is None:
        # set boundary conditions at r -> oo (u(oo) = 0 is implicit)
        u[-1] = 1.0

        # perform backwards integration from infinity to the turning point
        g = len(u) - 2
        u[-2] = u[-1] * f0[-1] / fm[-1]
        while c0[g] > 0.0:  # this defines the classical turning point
            u[g - 1] = (f0[g] * u[g] + fp[g] * u[g + 1]) / fm[g]
            if u[g - 1] < 0.0:
                raise RuntimeError(
                    "There should't be a node here!  Use a more negative eigenvalue"
                )
            if u[g - 1] > 1e100:
                u *= 1e-100
            g -= 1

        # stored values of the wavefunction and the first derivative
        # at the turning point
        gtp = g + 1
        utp = u[gtp]
        if gtp == len(u) - 1:
            return 100, 0.0
        dudrplus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    else:
        gtp = gmax

    # set boundary conditions at r -> 0
    u[0] = 0.0
    u[1] = 1.0

    # perform forward integration from zero to the turning point
    g = 1
    nodes = 0
    # integrate one step further than gtp
    # (such that dudr is defined in gtp)
    while g <= gtp:
        u[g + 1] = (fm[g] * u[g - 1] - f0[g] * u[g]) / fp[g]
        if u[g + 1] * u[g] < 0:
            nodes += 1
        g += 1
    if gmax is not None:
        return

    # scale first part of wavefunction, such that it is continuous at gtp
    u[: gtp + 2] *= utp / u[gtp]

    # determine size of the derivative discontinuity at gtp
    dudrminus = 0.5 * (u[gtp + 1] - u[gtp - 1]) / dr[gtp]
    A = (dudrplus - dudrminus) * utp

    return nodes, A
