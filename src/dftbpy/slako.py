from typing import Any

import numpy as np

from dftbpy.param.configs import angular_number
from dftbpy.spline import CubicSpline

slako_integral_types = {
    "ss": ["s"],
    "sp": ["s"],
    "ps": ["s"],
    "sd": ["s"],
    "ds": ["s"],
    "pp": ["s", "p"],
    "pd": ["s", "p"],
    "dp": ["s", "p"],
    "dd": ["s", "p", "d"],
}


slako_integrals = {
    "dds": 0,
    "ddp": 1,
    "ddd": 2,
    "pds": 3,
    "pdp": 4,
    "pps": 5,
    "ppp": 6,
    "sds": 7,
    "sps": 8,
    "sss": 9,
    "dps": 10,
    "dpp": 11,
    "dss": 12,
    "pss": 13,
}


class SlaterKosterSpline(CubicSpline):
    """Slater-Koster table cubic interpolator."""

    def __init__(self, R, table) -> None:
        self.n = table.shape[0]
        self.m = table.shape[1] // 2
        super().__init__(R, table)

    def get_cutoff(self):
        """Find cutoff distance for atom pair."""
        # Find max distance of lowest decaying sk hamitonian.
        h_ij = abs(self.y[:, : self.m])
        hcut = h_ij.max(0) * 1e-7
        i = self.n - 1
        while all(h_ij[i, :] < hcut):
            i -= 1
        return self.x[i]

    def __call__(self, x):
        """Given a value x, returns the cubic spline interpolated value y."""
        if not np.isscalar(x):
            # TODO!
            raise NotImplementedError("Can only accept scalars at the moment.")
        y, dy = super().__call__(x)
        h, s = y[: self.m], y[self.m :]
        dh, ds = dy[: self.m], dy[self.m :]
        return h, s, dh, ds


class SlaterKosterPair:
    """Slater-Koster table for a given atom pair."""

    def __init__(self, slako) -> None:
        s1, s2 = next(iter(slako.pairs))
        skt12 = slako.tables[s1, s2]
        skt21 = slako.tables[s2, s1]

        # dict of columns for overlap
        cols = {}
        for n1, l1 in slako.atoms[s1].get_valence_states():
            for n2, l2 in slako.atoms[s2].get_valence_states():
                for itype in slako_integral_types[l1 + l2]:
                    # 0 .., 13
                    ski = slako_integrals[l1 + l2 + itype]
                    if ski < 9:
                        hs = skt12[:, [ski, ski + 10]]
                    else:
                        # map to 0 .., 10
                        parity = (-1) ** (angular_number[l1] + angular_number[l2])
                        ski_ = slako_integrals[l2 + l1 + itype]
                        hs = parity * skt21[:, [ski_, ski_ + 10]]
                    # append
                    # index : {
                    # [[h(r1) ... s(r1)]
                    #         ...
                    # [h(rN)  ... s(rN)]]
                    # }
                    cols[ski] = hs

        n, m = len(slako.R), len(cols)
        table = np.empty((n, 2 * m))
        self.skindices = list(sorted(cols.keys()))
        for i, ski in enumerate(self.skindices):
            table[:, i] = cols[ski][:, 0]  # Hamiltonian
            table[:, i + m] = cols[ski][:, 1]  # Overlap

        self.table = SlaterKosterSpline(slako.R, table)
        self.cutoff = self.table.get_cutoff()

        # Internal
        self.h = np.zeros(14)
        self.s = np.zeros(14)
        self.dh = np.zeros((14, 3))
        self.ds = np.zeros((14, 3))

        no1 = slako.atoms[s1].get_number_of_valence_orbitals()
        no2 = slako.atoms[s2].get_number_of_valence_orbitals()
        self.tranform = SlaterKosterTransform(no1, no2)

    def __call__(self, rhat, dist) -> Any:
        h, s, dh, ds = self.table(dist)
        self.h[self.skindices] = h
        self.s[self.skindices] = s
        self.dh[self.skindices] = np.outer(dh, rhat)
        self.ds[self.skindices] = np.outer(ds, rhat)
        return self.tranform(rhat, dist, self.h, self.s, self.dh, self.ds)


class SlaterKosterTransform:
    s3 = np.sqrt(3.0)

    def __init__(self, no1, no2) -> None:
        self.no1 = no1
        self.no2 = no2

        self.mat = np.empty((9, 9, 14))
        self.ind = np.zeros((9, 9, 14), dtype=int)
        self.der = np.zeros((9, 9, 14, 3))
        self.cnt = np.zeros((9, 9), dtype=int) + 1
        self.cnt[1:, 1:] = 2
        self.cnt[4:, 4:] = 3

        self.ht = np.zeros((no1, no2))
        self.st = np.zeros((no1, no2))
        self.dht = np.zeros((no1, no2, 3))
        self.dst = np.zeros((no1, no2, 3))
        self.mxorb = max(no1, no2)

    def __call__(self, rhat, dist, h, s, dh, ds) -> Any:
        """
        Apply Slater-Koster transformation rules to orbitals iorbs and orbitals jorbs,
        where rhat is vector i->j and table gives the values for given tabulated
        matrix elements. Convention: orbital name starts with s,p,d,...
        """
        s3 = self.s3
        no1 = self.no1
        no2 = self.no2
        mat = self.mat
        ind = self.ind
        der = self.der
        cnt = self.cnt
        ht = self.ht
        st = self.st
        dht = self.dht
        dst = self.dst
        mxorb = self.mxorb

        x, y, z = rhat
        ll, mm, nn = rhat**2
        dx = (np.array([1, 0, 0]) - x * rhat) / dist
        dy = (np.array([0, 1, 0]) - y * rhat) / dist
        dz = (np.array([0, 0, 1]) - z * rhat) / dist
        dxx, dyy, dzz = 2 * x * dx, 2 * y * dy, 2 * z * dz

        mat[0, 0, 0] = 1  # ss
        der[0, 0, 0] = 0
        ind[0, 0, 0] = 9

        if mxorb >= 2:  # sp
            mat[0, 1, 0] = x
            der[0, 1, 0, :] = dx
            ind[0, 1, 0] = 8

            mat[0, 2, 0] = y
            der[0, 2, 0, :] = dy
            ind[0, 2, 0] = 8

            mat[0, 3, 0] = z
            der[0, 3, 0, :] = dz
            ind[0, 3, 0] = 8

        if mxorb >= 5:  # sd
            mat[0, 4, 0] = s3 * x * y
            der[0, 4, 0, :] = s3 * (dx * y + x * dy)
            ind[0, 4, 0] = 7

            mat[0, 5, 0] = s3 * y * z
            der[0, 5, 0, :] = s3 * (dy * z + y * dz)
            ind[0, 5, 0] = 7

            mat[0, 6, 0] = s3 * z * x
            der[0, 6, 0, :] = s3 * (dz * x + z * dx)
            ind[0, 6, 0] = 7

            mat[0, 7, 0] = 0.5 * s3 * (ll - mm)
            der[0, 7, 0, :] = 0.5 * s3 * (dxx - dyy)
            ind[0, 7, 0] = 7

            mat[0, 8, 0] = nn - 0.5 * (ll + mm)
            der[0, 8, 0, :] = dzz - 0.5 * (dxx + dyy)
            ind[0, 8, 0] = 7

        if mxorb >= 2:  # pp
            mat[1, 1, 0:2] = [ll, 1 - ll]
            der[1, 1, 0:2, :] = [dxx, -dxx]
            ind[1, 1, 0:2] = [5, 6]

            mat[1, 2, 0:2] = [x * y, -x * y]
            der[1, 2, 0:2, :] = [dx * y + x * dy, -(dx * y + x * dy)]
            ind[1, 2, 0:2] = [5, 6]

            mat[1, 3, 0:2] = [x * z, -x * z]
            der[1, 3, 0:2, :] = [dx * z + x * dz, -(dx * z + x * dz)]
            ind[1, 3, 0:2] = [5, 6]

        if mxorb >= 5:  # pd
            mat[1, 4, 0:2] = [s3 * ll * y, y * (1 - 2 * ll)]
            der[1, 4, 0:2, :] = [
                s3 * (dxx * y + ll * dy),
                dy * (1 - 2 * ll) + y * (-2 * dxx),
            ]
            ind[1, 4, 0:2] = [3, 4]

            mat[1, 5, 0:2] = [s3 * x * y * z, -2 * x * y * z]
            der[1, 5, 0:2, :] = [
                s3 * (dx * y * z + x * dy * z + x * y * dz),
                -2 * (dx * y * z + x * dy * z + x * y * dz),
            ]
            ind[1, 5, 0:2] = [3, 4]

            mat[1, 6, 0:2] = [s3 * ll * z, z * (1 - 2 * ll)]
            der[1, 6, 0:2, :] = [
                s3 * (dxx * z + ll * dz),
                dz * (1 - 2 * ll) + z * (-2 * dxx),
            ]
            ind[1, 6, 0:2] = [3, 4]

            mat[1, 7, 0:2] = [0.5 * s3 * x * (ll - mm), x * (1 - ll + mm)]
            der[1, 7, 0:2, :] = [
                0.5 * s3 * (dx * (ll - mm) + x * (dxx - dyy)),
                dx * (1 - ll + mm) + x * (-dxx + dyy),
            ]
            ind[1, 7, 0:2] = [3, 4]

            mat[1, 8, 0:2] = [x * (nn - 0.5 * (ll + mm)), -s3 * x * nn]
            der[1, 8, 0:2, :] = [
                dx * (nn - 0.5 * (ll + mm)) + x * (dzz - 0.5 * (dxx + dyy)),
                -s3 * (dx * nn + x * dzz),
            ]
            ind[1, 8, 0:2] = [3, 4]

        if mxorb >= 2:
            mat[2, 2, 0:2] = [mm, 1 - mm]
            der[2, 2, 0:2, :] = [dyy, -dyy]
            ind[2, 2, 0:2] = [5, 6]

            mat[2, 3, 0:2] = [y * z, -y * z]
            der[2, 3, 0:2, :] = [dy * z + y * dz, -(dy * z + y * dz)]
            ind[2, 3, 0:2] = [5, 6]

        if mxorb >= 5:
            mat[2, 4, 0:2] = [s3 * mm * x, x * (1 - 2 * mm)]
            der[2, 4, 0:2, :] = [
                s3 * (dyy * x + mm * dx),
                dx * (1 - 2 * mm) + x * (-2 * dyy),
            ]
            ind[2, 4, 0:2] = [3, 4]

            mat[2, 5, 0:2] = [s3 * mm * z, z * (1 - 2 * mm)]
            der[2, 5, 0:2, :] = [
                s3 * (dyy * z + mm * dz),
                dz * (1 - 2 * mm) + z * (-2 * dyy),
            ]
            ind[2, 5, 0:2] = [3, 4]

            mat[2, 6, 0:2] = [s3 * y * z * x, -2 * y * z * x]
            der[2, 6, 0:2, :] = [
                s3 * (dy * z * x + y * dz * x + y * z * dx),
                -2 * (dy * z * x + y * dz * x + y * z * dx),
            ]
            ind[2, 6, 0:2] = [3, 4]

            mat[2, 7, 0:2] = [0.5 * s3 * y * (ll - mm), -y * (1 + ll - mm)]
            der[2, 7, 0:2, :] = [
                0.5 * s3 * (dy * (ll - mm) + y * (dxx - dyy)),
                -(dy * (1 + ll - mm) + y * (dxx - dyy)),
            ]
            ind[2, 7, 0:2] = [3, 4]

            mat[2, 8, 0:2] = [y * (nn - 0.5 * (ll + mm)), -s3 * y * nn]
            der[2, 8, 0:2, :] = [
                dy * (nn - 0.5 * (ll + mm)) + y * (dzz - 0.5 * (dxx + dyy)),
                -s3 * (dy * nn + y * dzz),
            ]
            ind[2, 8, 0:2] = [3, 4]

        if mxorb >= 2:
            mat[3, 3, 0:2] = [nn, 1 - nn]
            der[3, 3, 0:2, :] = [dzz, -dzz]
            ind[3, 3, 0:2] = [5, 6]

        if mxorb >= 5:
            mat[3, 4, 0:2] = [s3 * x * y * z, -2 * y * z * x]
            der[3, 4, 0:2, :] = [
                s3 * (dx * y * z + x * dy * z + x * y * dz),
                -2 * (dy * z * x + y * dz * x + y * z * dx),
            ]
            ind[3, 4, 0:2] = [3, 4]

            mat[3, 5, 0:2] = [s3 * nn * y, y * (1 - 2 * nn)]
            der[3, 5, 0:2, :] = [
                s3 * (dzz * y + nn * dy),
                dy * (1 - 2 * nn) + y * (-2 * dzz),
            ]
            ind[3, 5, 0:2] = [3, 4]

            mat[3, 6, 0:2] = [s3 * nn * x, x * (1 - 2 * nn)]
            der[3, 6, 0:2, :] = [
                s3 * (dzz * x + nn * dx),
                dx * (1 - 2 * nn) + x * (-2 * dzz),
            ]
            ind[3, 6, 0:2] = [3, 4]

            mat[3, 7, 0:2] = [0.5 * s3 * z * (ll - mm), -z * (ll - mm)]
            der[3, 7, 0:2, :] = [
                0.5 * s3 * (dz * (ll - mm) + z * (dxx - dyy)),
                -(dz * (ll - mm) + z * (dxx - dyy)),
            ]
            ind[3, 7, 0:2] = [3, 4]

            mat[3, 8, 0:2] = [z * (nn - 0.5 * (ll + mm)), s3 * z * (ll + mm)]
            der[3, 8, 0:2, :] = [
                dz * (nn - 0.5 * (ll + mm)) + z * (dzz - 0.5 * (dxx + dyy)),
                s3 * (dz * (ll + mm) + z * (dxx + dyy)),
            ]
            ind[3, 8, 0:2] = [3, 4]

        if mxorb >= 5:
            mat[4, 4, 0:3] = [3 * ll * mm, ll + mm - 4 * ll * mm, nn + ll * mm]
            der[4, 4, 0:3, :] = [
                3 * (dxx * mm + ll * dyy),
                dxx + dyy - 4 * (dxx * mm + ll * dyy),
                dzz + (dxx * mm + ll * dyy),
            ]
            ind[4, 4, 0:3] = [0, 1, 2]

            mat[4, 5, 0:3] = [3 * x * mm * z, x * z * (1 - 4 * mm), x * z * (mm - 1)]
            der[4, 5, 0:3, :] = [
                3 * (dx * mm * z + x * dyy * z + x * mm * dz),
                dx * z * (1 - 4 * mm) + x * dz * (1 - 4 * mm) + x * z * (-4 * dyy),
                dx * z * (mm - 1) + x * dz * (mm - 1) + x * z * dyy,
            ]
            ind[4, 5, 0:3] = [0, 1, 2]

            mat[4, 6, 0:3] = [3 * ll * y * z, y * z * (1 - 4 * ll), y * z * (ll - 1)]
            der[4, 6, 0:3, :] = [
                3 * (dxx * y * z + ll * dy * z + ll * y * dz),
                dy * z * (1 - 4 * ll) + y * dz * (1 - 4 * ll) + y * z * (-4 * dxx),
                dy * z * (ll - 1) + y * dz * (ll - 1) + y * z * dxx,
            ]
            ind[4, 6, 0:3] = [0, 1, 2]

            mat[4, 7, 0:3] = [
                1.5 * x * y * (ll - mm),
                2 * x * y * (mm - ll),
                0.5 * x * y * (ll - mm),
            ]
            der[4, 7, 0:3, :] = [
                1.5 * (dx * y * (ll - mm) + x * dy * (ll - mm) + x * y * (dxx - dyy)),
                2 * (dx * y * (mm - ll) + x * dy * (mm - ll) + x * y * (dyy - dxx)),
                0.5 * (dx * y * (ll - mm) + x * dy * (ll - mm) + x * y * (dxx - dyy)),
            ]
            ind[4, 7, 0:3] = [0, 1, 2]

            mat[4, 8, 0:3] = [
                s3 * x * y * (nn - 0.5 * (ll + mm)),
                -2 * s3 * x * y * nn,
                0.5 * s3 * x * y * (1 + nn),
            ]
            der[4, 8, 0:3, :] = [
                s3
                * (
                    dx * y * (nn - 0.5 * (ll + mm))
                    + x * dy * (nn - 0.5 * (ll + mm))
                    + x * y * (dzz - 0.5 * (dxx + dyy))
                ),
                -2 * s3 * (dx * y * nn + x * dy * nn + x * y * dzz),
                0.5 * s3 * (dx * y * (1 + nn) + x * dy * (1 + nn) + x * y * dzz),
            ]
            ind[4, 8, 0:3] = [0, 1, 2]

            mat[5, 5, 0:3] = [3 * mm * nn, (mm + nn - 4 * mm * nn), (ll + mm * nn)]
            der[5, 5, 0:3, :] = [
                3 * (dyy * nn + mm * dzz),
                (dyy + dzz - 4 * (dyy * nn + mm * dzz)),
                (dxx + dyy * nn + mm * dzz),
            ]
            ind[5, 5, 0:3] = [0, 1, 2]

            mat[5, 6, 0:3] = [3 * y * nn * x, y * x * (1 - 4 * nn), y * x * (nn - 1)]
            der[5, 6, 0:3, :] = [
                3 * (dy * nn * x + y * dzz * x + y * nn * dx),
                dy * x * (1 - 4 * nn) + y * dx * (1 - 4 * nn) + y * x * (-4 * dzz),
                dy * x * (nn - 1) + y * dx * (nn - 1) + y * x * dzz,
            ]
            ind[5, 6, 0:3] = [0, 1, 2]

            mat[5, 7, 0:3] = [
                1.5 * y * z * (ll - mm),
                -y * z * (1 + 2 * (ll - mm)),
                y * z * (1 + 0.5 * (ll - mm)),
            ]
            der[5, 7, 0:3, :] = [
                1.5 * (dy * z * (ll - mm) + y * dz * (ll - mm) + y * z * (dxx - dyy)),
                -(
                    dy * z * (1 + 2 * (ll - mm))
                    + y * dz * (1 + 2 * (ll - mm))
                    + y * z * (2 * dxx - 2 * dyy)
                ),
                dy * z * (1 + 0.5 * (ll - mm))
                + y * dz * (1 + 0.5 * (ll - mm))
                + y * z * (0.5 * (dxx - dyy)),
            ]
            ind[5, 7, 0:3] = [0, 1, 2]

            mat[5, 8, 0:3] = [
                s3 * y * z * (nn - 0.5 * (ll + mm)),
                s3 * y * z * (ll + mm - nn),
                -0.5 * s3 * y * z * (ll + mm),
            ]
            der[5, 8, 0:3, :] = [
                s3
                * (
                    dy * z * (nn - 0.5 * (ll + mm))
                    + y * dz * (nn - 0.5 * (ll + mm))
                    + y * z * (dzz - 0.5 * (dxx + dyy))
                ),
                s3
                * (
                    dy * z * (ll + mm - nn)
                    + y * dz * (ll + mm - nn)
                    + y * z * (dxx + dyy - dzz)
                ),
                -0.5
                * s3
                * (dy * z * (ll + mm) + y * dz * (ll + mm) + y * z * (dxx + dyy)),
            ]
            ind[5, 8, 0:3] = [0, 1, 2]

            mat[6, 6, 0:3] = [3 * nn * ll, (nn + ll - 4 * nn * ll), (mm + nn * ll)]
            der[6, 6, 0:3, :] = [
                3 * (dzz * ll + nn * dxx),
                dzz + dxx - 4 * (dzz * ll + nn * dxx),
                (dyy + dzz * ll + nn * dxx),
            ]
            ind[6, 6, 0:3] = [0, 1, 2]

            mat[6, 7, 0:3] = [
                1.5 * z * x * (ll - mm),
                z * x * (1 - 2 * (ll - mm)),
                -z * x * (1 - 0.5 * (ll - mm)),
            ]
            der[6, 7, 0:3, :] = [
                1.5 * (dz * x * (ll - mm) + z * dx * (ll - mm) + z * x * (dxx - dyy)),
                dz * x * (1 - 2 * (ll - mm))
                + z * dx * (1 - 2 * (ll - mm))
                + z * x * (-2 * (dxx - dyy)),
                -(
                    dz * x * (1 - 0.5 * (ll - mm))
                    + z * dx * (1 - 0.5 * (ll - mm))
                    + z * x * (-0.5 * (dxx - dyy))
                ),
            ]
            ind[6, 7, 0:3] = [0, 1, 2]

            mat[6, 8, 0:3] = [
                s3 * x * z * (nn - 0.5 * (ll + mm)),
                s3 * x * z * (ll + mm - nn),
                -0.5 * s3 * x * z * (ll + mm),
            ]
            der[6, 8, 0:3, :] = [
                s3
                * (
                    dx * z * (nn - 0.5 * (ll + mm))
                    + x * dz * (nn - 0.5 * (ll + mm))
                    + x * z * (dzz - 0.5 * (dxx + dyy))
                ),
                s3
                * (
                    dx * z * (ll + mm - nn)
                    + x * dz * (ll + mm - nn)
                    + x * z * (dxx + dyy - dzz)
                ),
                -0.5
                * s3
                * (dx * z * (ll + mm) + x * dz * (ll + mm) + x * z * (dxx + dyy)),
            ]
            ind[6, 8, 0:3] = [0, 1, 2]

            mat[7, 7, 0:3] = [
                0.75 * (ll - mm) ** 2,
                (ll + mm - (ll - mm) ** 2),
                (nn + 0.25 * (ll - mm) ** 2),
            ]
            der[7, 7, 0:3, :] = [
                0.75 * 2 * (ll - mm) * (dxx - dyy),
                (dxx + dyy - 2 * (ll - mm) * (dxx - dyy)),
                (dzz + 0.25 * 2 * (ll - mm) * (dxx - dyy)),
            ]
            ind[7, 7, 0:3] = [0, 1, 2]

            mat[7, 8, 0:3] = [
                0.5 * s3 * (ll - mm) * (nn - 0.5 * (ll + mm)),
                s3 * nn * (mm - ll),
                0.25 * s3 * (1 + nn) * (ll - mm),
            ]
            der[7, 8, 0:3, :] = [
                0.5
                * s3
                * (
                    (dxx - dyy) * (nn - 0.5 * (ll + mm))
                    + (ll - mm) * (dzz - 0.5 * (dxx + dyy))
                ),
                s3 * (dzz * (mm - ll) + nn * (dyy - dxx)),
                0.25 * s3 * (dzz * (ll - mm) + (1 + nn) * (dxx - dyy)),
            ]
            ind[7, 8, 0:3] = [0, 1, 2]

            mat[8, 8, 0:3] = [
                (nn - 0.5 * (ll + mm)) ** 2,
                3 * nn * (ll + mm),
                0.75 * (ll + mm) ** 2,
            ]
            der[8, 8, 0:3, :] = [
                2 * (nn - 0.5 * (ll + mm)) * (dzz - 0.5 * (dxx + dyy)),
                3 * (dzz * (ll + mm) + nn * (dxx + dyy)),
                0.75 * 2 * (ll + mm) * (dxx + dyy),
            ]
            ind[8, 8, 0:3] = [0, 1, 2]

        # use the same rules for orbitals when they are reversed (pd ->dp)...
        for a in range(9):
            for b in range(a + 1, 9):
                mat[b, a, :] = mat[a, b, :]
                der[b, a, :, :] = der[a, b, :, :]
                ind[b, a, :] = ind[a, b, :]

        # ...but use different indices from table
        # pd 3:5-->10:12
        # sd 7->12
        # sp 8->13
        ind[1, 0, 0] = 13
        ind[2, 0, 0] = 13
        ind[3, 0, 0] = 13
        ind[4, 0, 0] = 12
        ind[5, 0, 0] = 12
        ind[6, 0, 0] = 12
        ind[7, 0, 0] = 12
        ind[8, 0, 0] = 12
        ind[4, 1, 0:2] = [10, 11]
        ind[5, 1, 0:2] = [10, 11]
        ind[6, 1, 0:2] = [10, 11]
        ind[7, 1, 0:2] = [10, 11]
        ind[8, 1, 0:2] = [10, 11]
        ind[4, 2, 0:2] = [10, 11]
        ind[5, 2, 0:2] = [10, 11]
        ind[6, 2, 0:2] = [10, 11]
        ind[7, 2, 0:2] = [10, 11]
        ind[8, 2, 0:2] = [10, 11]
        ind[4, 3, 0:2] = [10, 11]
        ind[5, 3, 0:2] = [10, 11]
        ind[6, 3, 0:2] = [10, 11]
        ind[7, 3, 0:2] = [10, 11]
        ind[8, 3, 0:2] = [10, 11]

        for i in range(no1):
            for j in range(no2):
                ht[i, j] = sum(
                    [mat[i, j, k] * h[ind[i, j, k]] for k in range(cnt[i, j])]
                )
                st[i, j] = sum(
                    [mat[i, j, k] * s[ind[i, j, k]] for k in range(cnt[i, j])]
                )
                for a in range(3):
                    dht[i, j, a] = sum(
                        [
                            mat[i, j, k] * dh[ind[i, j, k], a]
                            + der[i, j, k, a] * h[ind[i, j, k]]
                            for k in range(cnt[i, j])
                        ]
                    )
                    dst[i, j, a] = sum(
                        [
                            mat[i, j, k] * ds[ind[i, j, k], a]
                            + der[i, j, k, a] * s[ind[i, j, k]]
                            for k in range(cnt[i, j])
                        ]
                    )
        return ht, st, dht, dst
