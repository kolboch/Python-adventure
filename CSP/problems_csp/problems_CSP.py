from abc import ABC, abstractmethod

from CSP import AlgorithmCSP

import numpy as np


class ProblemCSP(ABC):
    @abstractmethod
    def __init__(self, problem_size, algorithm, heuristic):
        self.n = problem_size
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.matrix = None
        self.domain = None
        self.domains_matrix = None
        super().__init__()

    @abstractmethod
    def solve(self):
        raise NotImplementedError('This method should be implemented')

    def next_indices(self, x, y):
        if y + 1 < self.n:
            return x, y + 1
        elif x + 1 < self.n:
            return x + 1, 0
