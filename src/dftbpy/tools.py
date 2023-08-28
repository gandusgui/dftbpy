import numpy as np


def tri2full(A, UL="U", map=np.conj):
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
    N, tmp = A.shape
    assert N == tmp, "Matrix must be square"
    if UL != "L":
        A = A.T

    for n in range(N - 1):
        map(A[n + 1 :, n], A[n, n + 1 :])
