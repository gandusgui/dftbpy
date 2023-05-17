import numpy as np
from scipy.interpolate import splev, splint, splrep


class Spline:
    def __init__(self, x, y) -> None:
        self.a = x[0]
        self.b = x[-1]
        self.spl = splrep(x, y)

    def __call__(self, x, der=0):
        y = splev(x, self.spl, der=der)
        y = np.where(x < self.a, 0.0, y)
        y = np.where(x > self.b, 0.0, y)
        return y

    def intergrate(self):
        return splint(self.a, self.b, self.spl)

    def derivative(self, x):
        return self(x, der=1)

    def laplace(self, x):
        return self(x, der=2)
