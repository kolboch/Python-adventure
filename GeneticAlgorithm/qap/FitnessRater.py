from abc import ABC, abstractmethod
import numpy as np


class FitnessRater(ABC):
    """
        Interface for fitness raters
    """

    @abstractmethod
    def __init__(self, distances=None, flows=None):
        self.distances = distances
        self.flows = flows

    @abstractmethod
    def compute_scores(self, individuals):
        """
            accepts individuals of size NxM for whom computes scores,
            uses provided earlier distances MxM and flows MxM,
            returns N-length array of scores
        """
        raise NotImplementedError('This method must be implemented')


class ReversedFlowDistance(FitnessRater):
    """
        Implementation of FitnessRater.
        Compute score equation: 1 / (flow * distance)
    """

    def __init__(self, distances=None, flows=None):
        self.distances = distances
        self.flows = flows

    def compute_scores(self, individuals):
        if self.distances is None or self.flows is None:
            raise AttributeError('Distances and flows not provided.')
        scores = np.empty(individuals.shape[0])

        for i in range(0, individuals.shape[0]):
            score_buffer = 0
            for j in range(0, individuals.shape[1]):
                score_buffer += np.sum((self.distances[j] * self.flows[individuals[i][j]]))
            scores[i] = score_buffer / 2

        return 1 / scores
