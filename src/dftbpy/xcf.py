import numpy as np
from pylibxc import LibXCFunctional

# https://tddft.org/programs/libxc/functionals/


class LDA:
    """Slater exchange and Perdew-Wang correlation functional."""

    libxc_x_name = "LDA_X"
    libxc_c_name = "LDA_C_PW"

    def __init__(self) -> None:
        self.xcf = (
            LibXCFunctional(LDA.libxc_x_name, "unpolarized"),
            LibXCFunctional(LDA.libxc_c_name, "unpolarized"),
        )
        self.out = {"exc": None, "vxc": None}

    @property
    def exc(self):
        return np.squeeze(self.out["exc"])

    @property
    def vxc(self):
        return np.squeeze(self.out["vxc"])

    def compute(self, n):
        inp = {"rho": n}
        res = [f.compute(inp) for f in self.xcf]
        self.out = {
            "exc": sum(out["zk"] for out in res),
            "vxc": sum(out["vrho"] for out in res),
        }
