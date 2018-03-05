from abc import abstractmethod, ABC
import random


class Mutator(ABC):
    @abstractmethod
    def mutate(self, individual):
        """
        :param individual: to mutate
        :return: mutated individual
        """
        raise NotImplementedError('This method should be implemented')


class SwapMutator(Mutator):
    """
        swaps once or more single genes of given individual
        number of swaps is not greater than 1/8 individual's size
    """

    def mutate(self, individual):
        size = len(individual)
        for i in range(random.randint(0, int(size / 8))):
            a, b = random.sample(range(size), 2)
            individual[a], individual[b] = individual[b], individual[a]
        return individual


class ShuffleMutator(Mutator):
    """
        shuffles part of given individual, range is random
    """

    def mutate(self, individual):
        size = len(individual)
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a
        random.shuffle(individual[a:b])
        return individual


class InvertMutator(Mutator):
    """
        inverts random range of individual
    """

    def mutate(self, individual):
        size = len(individual)
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a
        individual[a:b] = individual[a:b][::-1]
        return individual
