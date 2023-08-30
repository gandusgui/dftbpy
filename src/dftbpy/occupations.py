from abc import abstractmethod
from typing import Callable, Tuple

import numpy as np


def fermi_dirac(
    eigs: np.ndarray, fermi_level: float, width: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Fermi-Dirac distribution function"""
    x = (eigs - fermi_level) / width
    x = np.clip(x, -100, 100)
    y = np.exp(x)
    z = y + 1.0
    f = 1.0 / z
    dfde = (f - f**2) / width  # derivative w.r.t. fermi level
    return f, dfde


class ZeroWidth:
    def __init__(self) -> None:
        self.width = 0.0

    def occupy(self, eigs, fermi_level):
        f = np.zeros_like(eigs)
        f[eigs < fermi_level] = 2.0
        f[eigs == fermi_level] = 1.0
        return f, 0.0

    def calculate_fermi_level(self, eigs, nel):
        """Occupy states."""
        no = eigs.size
        f = np.full(no, 2.0)
        cumf = np.cumsum(f)
        homo = np.searchsorted(cumf, nel)  # sum(cumf <= nel)
        if homo == no:
            fermi_level = np.inf
        extra = nel - cumf[homo]
        if extra > 0:
            assert extra <= 2.0
            f[homo] = extra
            fermi_level = eigs[homo]
        else:
            fermi_level = (eigs[homo + 1] + eigs[homo]) / 2
        return fermi_level


class Occupations(ZeroWidth):
    def __new__(cls, width):
        if width == 0.0:
            return ZeroWidth()

    def __init__(self, width=0.1) -> None:
        self.width = width

    @abstractmethod
    def occupy(self, eigs, fermi_level):
        ...

    def guess_fermi_level(self, eigs, nel):
        """Guess zero-width distribution."""
        return super().calculate_fermi_level(eigs, nel)

    def calculate_fermi_level(self, eigs, nel):
        fermi_level = self.guess_fermi_level(eigs, nel)

        if self.width == 0.0 or np.isinf(fermi_level):
            return fermi_level

        x = fermi_level

        def func(x):
            f, dfde = map(sum, self.occupy(eigs, x))
            df = f - nel
            return df, dfde

        fermi_level, niter = findroot(func, x)
        return fermi_level


class FermiDirac(Occupations):
    def occupy(self, eigs, fermi_level):
        f, dfde = fermi_dirac(eigs, fermi_level, self.width)
        f[:] *= 2.0
        dfde[:] *= 2.0
        return f, dfde


def findroot(
    func: Callable[[float], Tuple[float, float]], x: float, tol: float = 1e-10
) -> Tuple[float, int]:
    """Function used for locating Fermi level."""

    assert np.isfinite(x), x

    xmin = -np.inf
    xmax = np.inf

    # Try 10 step using the gradient:
    niter = 0
    while True:
        f, dfdx = func(x)
        if abs(f) < tol:
            return x, niter
        if f < 0.0 and x > xmin:
            xmin = x
        elif f > 0.0 and x < xmax:
            xmax = x
        dx = -f / max(dfdx, 1e-18)
        if niter == 10 or abs(dx) > 0.3 or not (xmin < x + dx < xmax):
            break  # try bisection
        x += dx
        niter += 1

    # Bracket the solution:
    if not np.isfinite(xmin):
        xmin = x
        fmin = f
        step = 0.01
        while fmin > tol:
            xmin -= step
            fmin = func(xmin)[0]
            step *= 2

    if not np.isfinite(xmax):
        xmax = x
        fmax = f
        step = 0.01
        while fmax < 0:
            xmax += step
            fmax = func(xmax)[0]
            step *= 2

    # Bisect:
    while True:
        x = (xmin + xmax) / 2
        f = func(x)[0]
        if abs(f) < tol:
            return x, niter
        if f > 0:
            xmax = x
        else:
            xmin = x
        niter += 1
        assert niter < 1000
