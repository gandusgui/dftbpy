from os import path

import numpy as np

from dftbpy.param.configs import angular_name, angular_number, valence_configurations
from dftbpy.param.slako import SlaterKosterTable

zeropattern = r"(\d+)\*(0\.0)"

energy_fields = [
    "Ed",
    "Ep",
    "Es",
    "SPE",
    "Ud",
    "Up",
    "Us",
    "fd",
    "fp",
    "fs",
]

extended_energy_fields = [
    "Ef",
    "Ed",
    "Ep",
    "Es",
    "SPE",
    "Uf",
    "Ud",
    "Up",
    "Us",
    "ff",
    "fd",
    "fp",
    "fs",
]


def is_extended(symbol):
    return any(angular_number[l] == 4 for n, l, f in valence_configurations[symbol])


class SKFile:
    def __init__(self, file) -> None:
        s1, s2 = path.basename(file).split(".")[0].split("-")
        # extended format
        self.is_extended = is_extended(s1)

        with open(file, "r") as fpt:
            self.fpt = fpt

            if s1 == s2:
                if self.is_extended:
                    self.skip_line()  # first comment line

                self.read_grid_line()
                self.read_energy_line()  # Energies

            else:
                if self.is_extended:
                    self.skip_line()

                self.read_grid_line()

            self.read_mass_line()  # Misc
            self.read_data_lines()

        self.table = {(s1, s2): self.data}

    def readline(self):
        return self.fpt.readline().replace(",", " ").split()

    def skip_line(self):
        self.readline()

    def read_grid_line(self):
        dr, N = self.readline()[:2]
        self.dr = float(dr)
        self.N = int(N)

    def read_energy_line(self):
        fields = extended_energy_fields if self.is_extended else energy_fields
        for name, value in zip(fields, self.readline()):
            setattr(self, name, float(value))

    def read_mass_line(self):
        self.skip_line()

    def read_data_lines(self):
        self.data = data = np.empty((self.N, 20))
        for iR in range(self.N):
            # read line
            j = 0
            for value in self.readline():
                # take car of 5*0.0 cases
                if "*" in value:
                    count, v = value.split("*")
                    count = int(count)
                    v = float(v)
                    for _ in range(count):
                        data[iR, j] = v
                        j += 1
                else:
                    v = float(value)
                    data[iR, j] = v
                    j += 1


# dR, N                                       distance, number of points
# Ed Ep Es N.A. Ud Up Us fd fp fs             energy, hubbard, cccupation
# -------------------------------
# Hdd0, Hdd1, Hdd2, .. Sdd0, Sdd1, ... Sss0   Hamiltonian, Overlap


class SKWrite:
    def __init__(self, slako: SlaterKosterTable, destdir="."):
        R = slako.R
        self.dR = R[1] - R[0]
        self.N = len(R)

        for s1, s2 in slako.pairs:
            # extended format
            self.is_extended = is_extended(s1)

            file = path.join(destdir, f"{s1}-{s2}.skf")

            self.data = slako.tables[(s1, s2)]
            self.a1 = slako.atoms[s1]
            self.a2 = slako.atoms[s2]

            with open(file, "w") as fpt:
                self.fpt = fpt

                if s1 == s2:
                    if self.is_extended:
                        self.skip_line()  # first comment line

                    self.write_grid_line()
                    self.write_energy_line()  # Energies

                else:
                    if self.is_extended:
                        self.skip_line()

                    self.write_grid_line()
                if s1 == s2:
                    self.write_mass_line()  # Misc
                self.write_data_lines()

    def printline(self, line):
        print(line, file=self.fpt)

    def skip_line(self):
        self.printline("")

    def write_grid_line(self):
        self.printline(f"{self.dR} {self.N}")

    def write_energy_line(self):
        fields = extended_energy_fields if self.is_extended else energy_fields
        values = dict.fromkeys(fields, 0.0)
        nl_j = self.a1.valence_configuration
        for nl in nl_j:
            j = self.a1.index(nl)
            l = angular_name[nl[1]]
            values["E" + l] = float(self.a1.e_j[j])
            values["f" + l] = float(self.a1.f_j[j])
        self.printline(" ".join([str(values[name]) for name in fields]))

    def write_mass_line(self):
        self.skip_line()

    def write_data_lines(self):
        # def squeeze(line):
        #     # squeeze 0.0
        #     out = []
        #     count = 0
        #     for num in line:
        #         if abs(num - 0.0) < 1e-7:
        #             count += 1
        #         else:
        #             if count > 0:
        #                 out.append(f"{count}*0.0")
        #                 count = 0
        #             out.append(str(num))
        #     if count > 0:
        #         out.append(f"{count}*0.0")
        #     return " ".join(out)

        # for line in self.data:
        #     self.printline(squeeze(line))

        np.savetxt(self.fpt, self.data, fmt="%.6e")
