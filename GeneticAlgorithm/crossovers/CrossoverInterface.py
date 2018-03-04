from abc import ABC, abstractmethod


class Crossover(ABC):
    @abstractmethod
    def crossover(self, individual_1, individual_2):
        raise NotImplementedError('This method must be implemented')
