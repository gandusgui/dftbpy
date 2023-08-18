import numpy as np


class CubicSpline:
    def __init__(self, x, y) -> None:
        self.x = np.array(x)  # shape = n
        self.y = np.array(y)  # shape = (n,m)
        self.y2 = np.empty_like(self.y)
        self.h = x[1] - x[0]
        self.a, self.b = x[0], x[-1]
        self.initialize()  # set second derivatives

    def initialize(self):
        """This routine stores an array y2[0..n-1] with second derivatives."""
        x = self.x
        y = self.y
        n, m = y.shape
        u = np.empty((n, m))  # temp
        y2 = self.y2
        u[0] = y2[0] = 0.0  # natural spline

        for i in range(1, n - 1):
            sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])
            p = sig * y2[i - 1] + 2.0
            y2[i] = (sig - 1.0) / p
            u[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - (y[i] - y[i - 1]) / (
                x[i] - x[i - 1]
            )
            u[i] = (6.0 * u[i] / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p
        qn = un = 0.0  # natural spline
        y2[n - 1] = (un - qn * u[n - 2]) / (qn * y2[n - 2] + 1.0)
        for k in range(n - 2, -1, -1):
            y2[k] = y2[k] * y2[k + 1] + u[k]

    def __call__(self, x):
        """Given a value x, returns the cubic spline interpolated value y."""
        xv = self.x
        yv = self.y
        y2 = self.y2
        h = self.h
        klo = int(((x - xv[0]) / (xv[-1] - xv[0])) * (len(xv) - 1))
        khi = klo + 1
        a = (xv[khi] - x) / h
        b = (x - xv[klo]) / h
        y = (
            a * yv[klo]
            + b * yv[khi]
            + ((a**3 - a) * y2[klo] + (b**3 - b) * y2[khi]) * (h**2) / 6.0
        )
        dy = (
            (yv[khi] - yv[klo]) / h
            - (3 * a**2 - 1) / 6 * h * y2[klo]
            + (3 * b**2 - 1) / 6 * h * y2[khi]
        )
        return y, dy
