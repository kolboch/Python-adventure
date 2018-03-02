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
        individuals_number = individuals.shape[0]
        problem_size = individuals.shape[1]
        for i in range(0, individuals_number):
            score_buffer = 0
            current = individuals[i]
            for j in range(0, problem_size):  # A, B, C, ...
                for k in range(j + 1, problem_size):  # AB, AC, AD ...
                    score_buffer += self.distances[j][k] * self.flows[current[j]][current[k]]
            scores[i] = score_buffer
        return scores
