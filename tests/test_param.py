# import pytest

import numpy as np

from dftbpy.param import Atom, SlaterKosterTable

__author__ = "ggandus"
__copyright__ = "ggandus"
__license__ = "MIT"


def test_atom():
    """API Tests"""
    atom = Atom("O")
    atom.run()
    np.testing.assert_allclose(
        atom.e_j, [-18.7576971944429, -0.8711995009080309, -0.3381738724978415]
    )


def test_slako():
    atom = Atom("O")
    atom.run()
    atom.v /= 2.0  # half the potential, H should give e_j
    slako = SlaterKosterTable(atom)
    slako.run(0.0, 0.2, 3)
    table = slako.skt[("O", "O")]
    # precision to 1e-4 [Hartee]
    assert abs(table[0, 5] - atom.e_j[2]) < 1e-4  # p integral
    assert abs(table[0, 6] - atom.e_j[2]) < 1e-4  # p integral
    assert abs(table[0, 9] - atom.e_j[1]) < 1e-4  # s integral


if __name__ == "__main__":
    test_atom()
    test_slako()

    # assert fib(1) == 1
    # assert fib(2) == 1
    # assert fib(7) == 13
    # with pytest.raises(AssertionError):
    #     fib(-10)


# def test_main(capsys):
#     """CLI Tests"""
#     # capsys is a pytest fixture that allows asserts against stdout/stderr
#     # https://docs.pytest.org/en/stable/capture.html
#     main(["7"])
#     captured = capsys.readouterr()
#     assert "The 7-th Fibonacci number is 13" in captured.out
