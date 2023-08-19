import numpy as np


def tri2full(H_nn, UL="U", map=np.conj):
    """Fill in values of hermitian or symmetric matrix.

    Fill values in lower or upper triangle of H_nn based on the opposite
    triangle, such that the resulting matrix is symmetric/hermitian.

    UL='U' will copy (conjugated) values from upper triangle into the
    lower triangle.

    UL='L' will copy (conjugated) values from lower triangle into the
    upper triangle.

    The map parameter can be used to specify a different operation than
    conjugation, which should work on 1D arrays.  Example::

      def antihermitian(src, dst):
            np.conj(-src, dst)

      tri2full(H_nn, map=antihermitian)

    """
    N, tmp = H_nn.shape
    assert N == tmp, "Matrix must be square"
    if UL != "L":
        H_nn = H_nn.T

    for n in range(N - 1):
        map(H_nn[n + 1 :, n], H_nn[n, n + 1 :])
