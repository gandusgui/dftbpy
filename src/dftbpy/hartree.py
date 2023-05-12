from math import pi


def hartree(l, nrdr, r, vr):
    """
    Hartree potential for a given density.
    """
    M = len(nrdr)
    p = 0.0
    q = 0.0
    for g in range(M - 1, 0, -1):
        R = r[g]
        rl = R**l
        dp = nrdr[g] / rl
        rlp1 = rl * R
        dq = nrdr[g] * rlp1
        vr[g] = (p + 0.5 * dp) * rlp1 - (q + 0.5 * dq) / rl
        p += dp
        q += dq
    vr[0] = 0.0
    f = 4.0 * pi / (2 * l + 1)
    for g in range(1, M):
        R = r[g]
        vr[g] = f * (vr[g] + q / R**l)
    return vr
