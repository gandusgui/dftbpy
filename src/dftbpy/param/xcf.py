from math import pi, sqrt

import numpy as np
from pylibxc import LibXCFunctional

# https://tddft.org/programs/libxc/functionals/


class LDA:
    """Slater exchange and Perdew-Wang correlation functional."""

    libxc_x_name = "LDA_X"
    libxc_c_name = "LDA_C_PW"

    def __init__(self, rgd=None) -> None:
        self.xcf = (
            LibXCFunctional(LDA.libxc_x_name, "unpolarized"),
            LibXCFunctional(LDA.libxc_c_name, "unpolarized"),
        )
        self.out = {"exc": None, "vxc": None}
        self.rgd = rgd

    @property
    def exc(self):
        # Units: Hartree
        return self.out["exc"]

    @property
    def vxc(self):
        return self.out["vxc"]

    @property
    def vrxc(self):
        return self.vxc * self.rgd.r_g

    def compute(self, n):
        inp = {"rho": n}
        res = [f.compute(inp) for f in self.xcf]
        self.out = {
            "exc": sum(out["zk"].reshape(-1) for out in res),
            "vxc": sum(out["vrho"].reshape(-1) for out in res),
        }


class PBE:
    """Perdew, Burke, Ernzerhof."""

    libxc_x_name = "GGA_X_PBE"
    libxc_c_name = "GGA_C_PBE"

    def __init__(self, rgd) -> None:
        self.xcf = (
            LibXCFunctional(PBE.libxc_x_name, "unpolarized"),
            LibXCFunctional(PBE.libxc_c_name, "unpolarized"),
        )
        self.out = {"exc": None, "vxc": None}
        self.rgd = rgd

    @property
    def exc(self):
        # Units: Hartree
        return self.out["exc"]

    @property
    def vxc(self):
        return self.out["vxc"]

    @property
    def vrxc(self):
        return self.vxc * self.rgd.r_g

    def compute(self, n):
        dndr = self.rgd.derivative(n)
        inp = {"rho": n, "sigma": dndr * dndr}
        res = [f.compute(inp) for f in self.xcf]
        self.out = {
            "exc": sum(out["zk"].reshape(-1) for out in res),
            "vxc": sum(out["vrho"].reshape(-1) for out in res),
        }
        vsigma = sum(out["vsigma"].reshape(-1) for out in res)
        self.out["vrxc"] = self.out["vxc"] * self.rgd.r_g - self.correction(
            vsigma, dndr
        )

    def correction(self, vsigma, dndr):
        dedg = 2.0 * vsigma * dndr
        d2edrdg = self.rgd.derivative(dedg)
        return 2.0 * dedg + d2edrdg * self.rgd.r_g


class PW92:
    def __init__(self, rgd=None):
        """The Perdew-Wang 1992 LDA exchange-correlation functional."""
        self.small = 1e-90
        self.a1 = 0.21370
        self.c0 = 0.031091
        self.c1 = 0.046644
        self.b1 = 1.0 / 2.0 / self.c0 * np.exp(-self.c1 / 2.0 / self.c0)
        self.b2 = 2 * self.c0 * self.b1**2
        self.b3 = 1.6382
        self.b4 = 0.49294
        self.out = {"exc": None, "vxc": None}
        self.rgd = rgd

    def calculate_exc(self, n, der=0):
        """Exchange-correlation with electron density n."""
        n = np.array(n)
        if n.ndim == 0:
            exc_clip = self.clipped_exc(n, der=der)
        elif n.ndim == 1:
            exc_clip = np.zeros_like(n)
            for i_n, n_i in enumerate(n):
                exc_clip[i_n] = self.clipped_exc(n_i, der=der)
        else:
            msg = "Got density of unexpected dimensionality " + str(n.ndim)
            raise ValueError(msg)
        return exc_clip

    def clipped_exc(self, n, der=0):
        if n < self.small:
            return 0.0
        else:
            return self.calculate_ex(n, der=der) + self.calculate_ec(n, der=der)

    def calculate_ex(self, n, der=0):
        """Exchange."""
        if der == 0:
            return -3.0 / 4 * (3 * n / pi) ** (1.0 / 3)
        elif der == 1:
            return -3.0 / (4 * pi) * (3 * n / pi) ** (-2.0 / 3)

    def calculate_ec(self, n, der=0):
        """Correlation energy."""
        sqrtrs = (3.0 / (4 * pi * n)) ** (1.0 / 6)
        rs = sqrtrs * sqrtrs
        aux = (
            2
            * self.c0
            * (
                self.b1 * sqrtrs
                + self.b2 * rs
                + self.b3 * rs ** (3.0 / 2)
                + self.b4 * rs**2
            )
        )
        auxinv = 1.0 / aux
        if der == 0:
            return -2 * self.c0 * (1 + self.a1 * rs) * np.log(1 + auxinv)
        elif der == 1:
            aux_new = 2 * sqrt(pi) * n * rs
            maux_new2inv = -1.0 / (aux_new * aux_new)
            return (
                -2 * self.c0 * self.a1 * np.log(1 + auxinv)
                - 2
                * self.c0
                * (1 + self.a1 * rs)
                * (1 + auxinv) ** -1
                * (-auxinv * auxinv)
                * 2
                * self.c0
                * (
                    self.b1 / (2 * sqrtrs)
                    + self.b2
                    + 3 * self.b3 * sqrtrs / 2
                    + 2 * self.b4 * rs
                )
            ) * maux_new2inv

    def calculate_vxc(self, n, exc, dexc):
        """Exchange-correlation potential (functional derivative of exc)."""
        return exc + n * dexc

    @property
    def exc(self):
        # Units: Hartree
        return self.out["exc"]

    @property
    def vxc(self):
        return self.out["vxc"]

    @property
    def vrxc(self):
        return self.vxc * self.rgd.r_g

    def compute(self, n):
        self.out["exc"] = exc = self.calculate_exc(n)
        dexc = self.calculate_exc(n, 1)
        self.out["vxc"] = self.calculate_vxc(n, exc, dexc)


xcf_names = {"lda": LDA, "pbe": PBE, "pw92": PW92}


def XC(name: str, rgd=None):
    return xcf_names[name.lower()](rgd)
