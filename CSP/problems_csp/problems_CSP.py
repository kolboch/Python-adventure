from abc import ABC, abstractmethod

from CSP import AlgorithmCSP

import numpy as np


class ProblemCSP(ABC):
    def __init__(self, problem_size, algorithm, heuristic):
        self.n = problem_size
        self.algorithm = algorithm
        self.heuristic = heuristic
        self.matrix = None
        self.default_value = -1
        super().__init__()

    @abstractmethod
    def solve(self):
        raise NotImplementedError('This method should be implemented')


class GraphColorL21(ProblemCSP):
    def solve(self):
        if self.algorithm == AlgorithmCSP.FORWARD_CHECKING:
            self.solve_forward_checking()
        elif self.algorithm == AlgorithmCSP.BACKTRACKING:
            self.solve_backtracking()
        else:
            raise ValueError('This value is not supported. Choose one of AlgorithmCSP values.')
        pass

    def solve_forward_checking(self):
        pass

    def solve_backtracking(self):
        colors_count = 1
        self.matrix = np.full(shape=(self.n, self.n), fill_value=-1, dtype=int)
        solved = False
        domains = np.full(shape=(self.n, self.n, colors_count), fill_value=[x for x in range(colors_count)], dtype=list)
        while not solved:
            colors_count += 1
            del domains[1, 1].tolist()[1]
            print(domains)
            solved = True

        print(self.matrix)

        pass

    # in every condition we check if it is already assigned, if not condition is satisfied
    def check_adjacent_diff_2(self, row, column):
        is_valid = True
        value = self.matrix[row, column]
        df = self.default_value
        # left adjacent
        if column - 1 >= 0:
            is_valid = self.matrix[row, column - 1] == df or np.absolute(self.matrix[row, column - 1] - value) >= 2
        # right adjacent
        if is_valid and column + 1 < self.matrix.shape[1]:
            is_valid = self.matrix[row, column + 1] == df or np.absolute(self.matrix[row, column + 1] - value) >= 2
        # top adjacent
        if is_valid and row - 1 >= 0:
            is_valid = self.matrix[row - 1, column] == df or np.absolute(self.matrix[row - 1, column] - value) >= 2
        # bottom adjacent
        if is_valid and row + 1 < self.matrix.shape[0]:
            is_valid = self.matrix[row + 1, column] == df or np.absolute(self.matrix[row + 1, column] - value) >= 2
        return is_valid

    # in every condition we check if it is already assigned, if not condition is satisfied
    def check_biases_diff_1(self, row, column):
        is_valid = True
        value = self.matrix[row, column]
        df = self.default_value
        # upper left
        if column - 1 >= 0 and row - 1 >= 0:
            is_valid = self.matrix[row - 1, column - 1] == df or np.absolute(self.matrix[row - 1, column - 1]) >= 1
        # upper right
        if is_valid and column + 1 <= self.matrix.shape[1] and row - 1 >= 0:
            is_valid = self.matrix[row - 1, column + 1] == df or np.absolute(self.matrix[row - 1, column + 1]) >= 1
        # lower left
        if is_valid and column - 1 >= 0 and row + 1 <= self.matrix.shape[0]:
            is_valid = self.matrix[row + 1, column - 1] == df or np.absolute(self.matrix[row + 1, column - 1]) >= 1
        # lower bottom
        if is_valid and column + 1 <= self.matrix.shape[1] and row + 1 <= self.matrix.shape[0]:
            is_valid = self.matrix[row + 1, column + 1] == df or np.absolute(self.matrix[row + 1, column + 1]) >= 1
        return is_valid


class LatinSquare(ProblemCSP):
    def solve(self):
        if self.algorithm == AlgorithmCSP.FORWARD_CHECKING:
            self.solve_forward_checking()
        elif self.algorithm == AlgorithmCSP.BACKTRACKING:
            self.solve_backtracking()
        else:
            raise ValueError('This value is not supported. Choose one of AlgorithmCSP values.')

    def solve_forward_checking(self):
        pass

    def solve_backtracking(self):
        pass
