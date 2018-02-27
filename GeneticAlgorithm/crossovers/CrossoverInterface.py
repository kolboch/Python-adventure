from abc import ABC, abstractmethod


class Crossover(ABC):

    @abstractmethod
    def crossover(self):
        raise NotImplementedError('This method must be implemented')
