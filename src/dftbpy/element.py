from dftbpy.param.configs import angular_number, atomic_numbers, valence_configurations


class Element:
    def __init__(self, symbol) -> None:
        self.symbol = symbol
        self.Z = atomic_numbers[symbol]

        Nv = 0  # number of valence electrons
        No = 0  # number of orbitals
        valence = {}
        for nlf in valence_configurations[symbol]:
            nl, f = nlf[:2], float(nlf[2])
            valence[nl] = {"e": 0, "f": f, "U": 0.0}
            Nv += f
            No += 2 * angular_number[nl[1]] + 1

        self.FWHM = 0.0  # full width at half maximum
        self.Nv = Nv
        self.No = No
        self.valence = valence  # valence electrons

    def get_valence_states(self):
        return self.valence.keys()
