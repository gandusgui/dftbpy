import pprint
from math import log, pi, sqrt
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.data import covalent_radii as covalent_radii_angstrom
from ase.units import Bohr

from dftbpy.param.configs import (
    configurations,
    convert_configuration,
    dissociation_energies,
    valence_configurations,
)
from dftbpy.param.hartree import hartree
from dftbpy.param.spline import Spline
from dftbpy.param.xcf import XC


def covalent_radii(symbol):
    return covalent_radii_angstrom[atomic_numbers[symbol]] / Bohr


class Grid:
    """r=ag/(1-bg), g=0,..,N-1"""

    def __init__(self, a, b, N) -> None:
        self.a = a
        self.b = b
        self.N = N
        self.g = np.arange(N)
        self.r_g = self.a * self.g / (1 - self.b * self.g)  # radial grid
        self.dr_g = (self.b * self.r_g + self.a) ** 2 / self.a  # dr/dg

    def empty(self):
        return np.empty(self.N)

    def zeros(self):
        a = self.empty()
        a[:] = 0.0
        return a

    def integrate(self, a_xg, n=0):
        assert n >= -2
        return np.dot(a_xg[..., 1:], (self.r_g ** (2 + n) * self.dr_g)[1:]) * (4 * pi)

    def d2gdr2(self):
        return -2 * self.a * self.b / (self.b * self.r_g + self.a) ** 3

    def derivative(self, n_g, dndr_g=None):
        """Finite-difference derivative of radial function."""
        if dndr_g is None:
            dndr_g = self.empty()
        dndr_g[0] = n_g[1] - n_g[0]
        dndr_g[1:-1] = 0.5 * (n_g[2:] - n_g[:-2])
        dndr_g[-1] = n_g[-1] - n_g[-2]
        dndr_g /= self.dr_g
        return dndr_g

    def poisson(self, n_g, l=0):
        vr_g = self.zeros()
        nrdr_g = n_g * self.r_g * self.dr_g
        hartree(l, nrdr_g, self.r_g, vr_g)
        return vr_g

    def kin(self, u_g, l):
        """Radial kinetic operator."""
        # tau = 0.
        # for l, f, u in zip(atom.l_j, atom.f_j, atom.u_j):
        #     tau += f * np.dot(u * kin(u, l), dr_g)
        dudg_g = 0.5 * (u_g[2:] - u_g[:-2])
        d2udg2_g = u_g[2:] - 2 * u_g[1:-1] + u_g[:-2]
        Tu_g = self.empty()
        Tu_g[1:-1] = -0.5 * (
            d2udg2_g / self.dr_g[1:-1] ** 2 + dudg_g * self.d2gdr2()[1:-1]
        )
        Tu_g[-1] = Tu_g[-2]
        Tu_g[1:] += 0.5 * l * (l + 1) * u_g[1:] / self.r_g[1:] ** 2
        Tu_g[0] = Tu_g[1]
        return Tu_g

    def kin2(self, R_g, l):
        # tau = 0.
        # for l, f, R in zip(atom.l_j, atom.f_j, atom.R_j):
        #     tau += f * atom.rgd.integrate(R * atom.rgd.kin2(R, l))
        # tau /= (4*np.pi)
        dRdg_g = 0.5 * (R_g[2:] - R_g[:-2])
        TR_g = self.kin(R_g, l)
        TR_g[1:-1] -= dRdg_g / (self.r_g[1:-1] * self.dr_g[1:-1])
        TR_g[0] = TR_g[1]
        TR_g[-1] = TR_g[-2]
        return TR_g

    def calculate_kinetic_energy_density(self, R_g, l):
        """Calculate kinetic energy density."""
        tau_g = self.derivative(R_g) ** 2 / (8 * pi)
        if l > 0:
            tau_g[1:] += l * (l + 1) * (R_g[1:] / self.r_g[1:]) ** 2 / (8 * pi)
        return tau_g

    def get_cutoff(self, f_g):
        f_g = abs(f_g)
        fcut = f_g.max() * 1e-7  # fractional limit
        g = self.N - 1
        while f_g[g] < fcut:
            g -= 1
        gcut = g + 1
        return gcut


# Helpers
def take_nl(nlf):
    return nlf[:2]


def compress_repr(conf):
    return list(map(take_nl, conf))


class Atom:
    def __init__(
        self,
        symbol,
        configuration=None,
        xc="lda",
        confinement={"mode": "none"},
        gpernode=150,
    ) -> None:
        self.symbol = symbol
        self.Z = atomic_numbers[symbol]
        if configuration is None:
            configuration = configurations[symbol]

        nlf_j = [convert_configuration(nlf) for nlf in configuration]
        self.n_j = [n for n, l, f in nlf_j]
        self.l_j = [l for n, l, f in nlf_j]
        self.f_j = [f for n, l, f in nlf_j]
        self.e_j = list(configuration.values())

        # grid
        maxnodes = max([n - l - 1 for n, l in zip(self.n_j, self.l_j)])
        self.N = (maxnodes + 1) * gpernode
        self.beta = 0.4
        self.rgd = Grid(self.beta / self.N, 1.0 / self.N, self.N)

        # xc
        self.xc = XC(xc, self.rgd)

        # confinement
        self.vconf = ConfinementPotential(**{**confinement, **dict(symbol=symbol)})

        # orbs
        self.nj = len(self.n_j)
        self.u_j = np.zeros((self.nj, self.N))  # radial wave functions times radius
        self.R_j = np.zeros((self.nj, self.N))  # radial wave functions
        self.vr = np.zeros(self.N)  # potential times radius
        self.v = np.zeros(self.N)  # potential
        self.n = np.zeros(self.N)  # electron density

        # energies
        self.Exc = 0.0
        self.Ecoul = 0.0
        self.Ekin = 0.0
        self.Enucl = 0.0
        self.Etot = 0.0

        self.reference = {
            "configuration": configurations[self.symbol],
            "valence_configuration": valence_configurations[self.symbol],
        }
        self.nupdates = 0

    @property
    def charge(self):
        return self.Z - sum(self.f_j)

    @property
    def nlfe_j(self):
        yield from zip(self.n_j, self.l_j, self.f_j, self.e_j)

    @property
    def configuration(self):
        """Dictionary of {nlf:e} configuration."""
        return {convert_configuration(n, l, f): e for n, l, f, e in self.nlfe_j}

    @property
    def valence_configuration(self):
        """Dictionary of {nlf:e} valence configuration."""
        ref_nl_conf = compress_repr(self.reference["configuration"])
        ref_nl_valence = compress_repr(self.reference["valence_configuration"])

        valence_configuration = {}
        for nlf, e in self.configuration.items():
            nl = take_nl(nlf)
            # add configuration to valence configuration
            # if present in reference valence configuration
            # (also with a different occupation)
            # or not present in original configuration
            if nl in ref_nl_valence or nl not in ref_nl_conf:
                valence_configuration[nlf] = e

        return valence_configuration

    def guess_radials(self):
        if self.nupdates > 0:
            return
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
        # sum_nl |u_nl(r)|**2/(4*pi*r^2)
        n = np.dot(self.f_j, np.where(abs(self.u_j) < 1e-160, 0, self.u_j) ** 2) / (
            4 * pi
        )
        n[1:] /= self.rgd.r_g[1:] ** 2
        n[0] = n[1]
        return n

    def solve_radials(self):
        rgd = self.rgd
        r = rgd.r_g
        dr = rgd.dr_g
        vr = self.vr

        c2 = -((r / dr) ** 2)
        c1 = -rgd.d2gdr2() * r**2
        # solve for each quantum state separately
        for j, (n, l, e, u) in enumerate(zip(self.n_j, self.l_j, self.e_j, self.u_j)):
            nodes = n - l - 1  # analytically expected number of nodes

            emax = vr[-1] / r[-1] - l * (l + 1) / (2 * r[-1] ** 2)
            delta = self.de_j[j]
            dir = ""
            niter = 0
            while True:
                eprev = e
                nn, A = shoot(u, l, vr, e, r, dr, c1, c2)
                niter += 1

                norm = rgd.integrate(u**2, -2) / (4 * pi)
                u *= 1.0 / sqrt(norm)
                if nn > nodes:
                    # decrease energy
                    if dir == "up":
                        delta /= 2
                    e -= delta
                    dir = "down"
                elif nn < nodes:
                    # increase energy
                    if dir == "down":
                        delta /= 2
                    e += delta
                    dir = "up"
                elif nn == nodes:
                    de = -0.5 * A / norm
                    if abs(de) < 1e-9:  # convergence
                        break
                    if de > 0:
                        dir = "up"
                    elif de < 0:
                        dir = "down"
                    e += de
                if e > emax:
                    e = 0.5 * (emax + eprev)

                assert niter < 400, (n, l, e)
            self.e_j[j] = e

    def run(self, nitermax=117, qOK=log(1e-10)):
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
        self.niter = 0
        # nitermax = 117
        # qOK = log(1e-10)
        mix = 0.4

        vrold = None
        vHr = np.zeros(self.N)

        self.de_j = -0.2 * np.array(self.e_j)

        while True:
            # harten potential
            hartree(0, n * r * dr, r, vHr)  # radial integration dr r^2
            # nuclear potential
            vHr -= self.Z
            # exchange-correlation potential
            self.xc.compute(n)
            vr[:] = vHr + self.xc.vrxc  # * r
            # confinement
            vr[:] += self.vconf(r) * r
            # mix
            if self.niter > 0:
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
            self.niter += 1
            if self.niter > nitermax:
                raise RuntimeError(
                    f"""
                    Maximum number of iterations exceeded!
                    Error is {q} while tolerance is {qOK}."""
                )

        self.e_j = np.array(self.e_j)
        # Energy contributions
        Ekin = 0.0
        for f, e in zip(self.f_j, self.e_j):
            Ekin += f * e

        self.Ecoul = 2 * pi * np.dot(n * r * (vHr + self.Z), dr)
        self.Ekin = Ekin - 4 * pi * np.dot(
            n * vr * r, dr
        )  # same as self.calculate_kinetic_energy_density() method
        self.Exc = self.rgd.integrate(self.xc.exc * self.n)
        self.Enucl = -4 * pi * np.dot(n * r * self.Z, dr)
        self.Etot = self.Exc + self.Ecoul + self.Ekin + self.Enucl

        # # Radial functions
        r1 = r[1]
        r2 = r[2]
        for l, R, u in zip(self.l_j, self.R_j, self.u_j):
            R[1:] = u[1:] / r[1:]
            if l == 0:
                # Extrapolation with midpoint formula.
                R[0] = 0.5 * (R[1] + R[2] + (R[1] - R[2]) * (r1 + r2) / (r2 - r1))
            else:
                R[0] = 0

        # Electronic potential
        v = self.v
        v[1:] = self.vr[1:] / r[1:]
        # Extrapolation with midpoint formula.
        v[0] = 0.5 * (v[1] + v[2] + (v[1] - v[2]) * (r1 + r2) / (r2 - r1))

        self.nupdates += 1

    def calculate_kinetic_energy_density(self):
        """Calculate kinetic energy density."""
        # Equivalent but more accurate than
        # tau = self.rgd.zeros()
        # for l, f, R in zip(self.l_j, self.f_j, self.R_j):
        #     tau += f * self.rgd.calculate_kinetic_energy_density(R, l)
        # self.rgd.integrate(tau)
        dudr = np.zeros(self.N)
        tau = np.zeros(self.N)
        r = self.rgd.r_g
        for f, l, u in zip(self.f_j, self.l_j, self.u_j):
            self.rgd.derivative(u, dudr)
            # contribution from angular derivatives
            if l > 0:
                tau += f * l * (l + 1) * np.where(abs(u) < 1e-160, 0, u) ** 2
            # contribution from radial derivatives
            dudr = u - r * dudr
            tau += f * np.where(abs(dudr) < 1e-160, 0, dudr) ** 2
        tau[1:] /= r[1:] ** 4
        tau[0] = tau[1]

        return 0.5 * tau / (4 * pi)

    def calculate_number_of_electrons(self):
        return self.rgd.integrate(self.n)

    def get_cutoff(self):
        gcut = max(self.rgd.get_cutoff(R) for R in self.R_j)
        return self.rgd.r_g[gcut]

    def index(self, nlf):
        """Return index of nl state."""
        n, l, f = convert_configuration(nlf)
        for j, (n_, l_) in enumerate(zip(self.n_j, self.l_j)):
            if n_ == n and l_ == l:
                return j
        raise RuntimeError("State not found.")

    def spline(self, x):
        return Spline(self.rgd.r_g, x)

    def todict(self):
        d = {}
        d["symbol"] = self.symbol
        d["configuration"] = self.configuration
        d["vconf"] = vconf = {"mode": self.vconf.mode}
        if vconf["mode"] != "none":
            vconf["r0"] = self.vconf.r0
        d["Exc"] = self.Exc
        d["Ecoul"] = self.Ecoul
        d["Ekin"] = self.Ekin
        d["Enucl"] = self.Enucl
        d["Etot"] = self.Etot
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


def get_confinement(mode, *p):
    def none(r):
        """Confinement potential function for the 'none' mode."""
        return 0.0

    def pow(r):
        """Confinement potential function for the 'frauenheim' mode."""
        return (r / p[0]) ** p[1]

    def morse(r):
        """Confinement potential function for the 'frauenheim' mode."""
        return p[0] * (1 + np.exp(-2 * p[1] * r) - 2 * np.exp(-p[1] * r))
        # return p[0] * (1 - np.exp(p[1] * r)) ** 2

    mode_functions = {
        "none": none,
        "pow": pow,
        "morse": morse,
    }

    return mode_functions[mode]


class ConfinementPotential:
    def __init__(self, *, mode, **kwargs):
        """
        Initialize the ConfinementPotential object.

        :param mode: The mode of the confinement potential.
        :param kwargs: Additional mode-specific arguments.
        """
        symbol = kwargs.get("symbol", None)
        r0 = kwargs.get("r0", 2 * covalent_radii(symbol))
        p = []
        if mode == "pow":
            a = kwargs.get("a", 2)
            p = [r0, a]
        elif mode == "morse":
            D = kwargs.get("D", dissociation_energies[symbol])
            a = np.sqrt(1 / D) / r0
            p = [D, a]
        self.f = get_confinement(mode, *p)

    def __call__(self, r):
        """Calculate the confinement potential for a given radius."""
        return self.f(r)

    def __str__(self):
        """Description for the current mode."""
        return f"ConfinementPotential(mode='{self.mode}', kwargs={self.extra})"


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

    u[:] *= np.sign(u[1])
    return nodes, A
