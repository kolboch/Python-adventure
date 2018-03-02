class Individual:
    def __init__(self, genotype=None):
        self._genotype = genotype
        pass

    @property
    def genotype(self):
        return self._genotype

    @genotype.setter
    def genotype(self, genotype):
        self._genotype = genotype

