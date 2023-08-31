from abc import abstractmethod

from dftbpy.setups import Setups


def arrayproperty(name, doc):
    """Helper function to create array property."""

    def getter(self):
        return self.arrays[name]

    def deleter(self):
        return self.arrays.pop(name)

    return property(getter, None, deleter, doc)


class SimpleCalculator:
    def __init__(self, setups: Setups) -> None:
        self.setups = setups

        self.arrays = {}
        self.metarrays = {}  # arrays' metadata
        self.nupdates = 0

    @abstractmethod
    def calculate(self, *args):
        ...

    @abstractmethod
    def requires_calculation(self, *args):
        ...

    @abstractmethod
    def update(self, *args):
        ...

    def set_arrays(self):
        for name, (atype, params) in self.metarrays.items():
            # Params:
            # - shape: tuple of ints and/or str, where int(str)=getattr(setups,str).
            # - same as np.ndarray parameters
            data = self.arrays.get(name)
            if data is None:  # not yet initialized
                shape = list(tuple(params.get("shape")))
                for i in range(len(shape)):
                    x = shape[i]
                    if isinstance(x, str):
                        shape[i] = getattr(self.setups, x)
                    else:  # int
                        shape[i] = int(x)
                data = atype(**{**params, **dict(shape=shape)})

            atype.fill(data, 0.0)

            self.arrays[name] = data


class SetupConsistent(SimpleCalculator):
    def requires_calculation(self):
        #
        setups = self.setups

        if self.nupdates == 0:
            assert setups.nupdates != 0, "Setups not initialized."

        if setups.nupdates != self.nupdates:
            return True

        return False  # no change

    def update(self, *args):
        #
        update = False
        if self.requires_calculation():
            self.set_arrays()
            self.calculate(*args)
            self.nupdates += 1  # up-to-date with setups
            update = True

        assert self.nupdates == self.setups.nupdates, "Self and Setups differ."
        return update
