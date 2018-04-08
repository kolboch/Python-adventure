from CSP import AlgorithmCSP
from problems_csp.problems_CSP import ProblemCSP
import numpy as np


class GraphColorL21(ProblemCSP):
    def __init__(self, problem_size, algorithm, heuristic):
        super().__init__(problem_size, algorithm, heuristic)
        self.colors_count = 0
        self.default_value = -1

    def solve(self):
        if self.algorithm == AlgorithmCSP.FORWARD_CHECKING:
            return self.solve_forward_checking()
        elif self.algorithm == AlgorithmCSP.BACKTRACKING:
            return self.solve_backtracking()
        else:
            raise ValueError('This value is not supported. Choose one of AlgorithmCSP values.')

    def solve_forward_checking(self):
        self.colors_count = 1
        self.matrix = np.full(shape=(self.n, self.n), fill_value=-1, dtype=int)
        solved = False
        while not solved:
            self.init_domains(self.colors_count)
            solved = self.forward_solve_util(0, 0)
            self.colors_count += 1
        return self.matrix

    def init_domains(self, max_colors):
        self.domains_matrix = [[[z for z in range(max_colors)] for _ in range(self.n)] for _ in range(self.n)]

    def forward_solve_util(self, x, y):
        for color in self.domains_matrix[x][y]:
            self.matrix[x, y] = color
            self.validate_domains_matrix(x, y, color)
            if (x == self.n - 1 and y == self.n - 1) or self.forward_solve_util(*(self.next_indices(x, y))):
                return True
            self.restore_domains_from(x, y)
        return False

    def restore_domains_from(self, row, column):
        # right adjacent
        if column + 1 < self.matrix.shape[1]:
            self.domains_matrix[row][column - 1] = [z for z in range(self.colors_count)]
        # bottom adjacent
        if row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row][column - 1] = [z for z in range(self.colors_count)]
        # lower left
        if column - 1 >= 0 and row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row][column - 1] = [z for z in range(self.colors_count)]
        # lower right
        if column + 1 < self.matrix.shape[1] and row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row][column - 1] = [z for z in range(self.colors_count)]

    def validate_domains_matrix(self, row, column, value):
        # right adjacent
        if column + 1 < self.matrix.shape[1]:
            self.domains_matrix[row][column + 1] = [item for item in self.domains_matrix[row][column + 1] if
                                                    np.absolute(value - item) >= 2]
        # bottom adjacent
        if row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row + 1][column] = [item for item in self.domains_matrix[row + 1][column] if
                                                    np.absolute(value - item) >= 2]
        # lower left
        if column - 1 >= 0 and row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row + 1][column - 1] = [item for item in self.domains_matrix[row + 1][column - 1] if
                                                        np.absolute(value - item) >= 2]
        # lower right
        if column + 1 < self.matrix.shape[1] and row + 1 < self.matrix.shape[0]:
            self.domains_matrix[row + 1][column + 1] = [item for item in self.domains_matrix[row + 1][column + 1] if
                                                        np.absolute(value - item) >= 2]

    def solve_backtracking(self):
        self.colors_count = 1
        self.matrix = np.full(shape=(self.n, self.n), fill_value=-1, dtype=int)
        solved = False
        while not solved:
            self.domain = np.arange(0, self.colors_count)
            solved = self.solve_util(0, 0)
            self.colors_count += 1
        return self.matrix

    def solve_util(self, x, y):
        for color in self.domain:
            if self.is_valid(x, y, color):
                self.matrix[x, y] = color
                if (x == self.n - 1 and y == self.n - 1) or self.solve_util(*(self.next_indices(x, y))):
                    return True
                self.matrix[x, y] = self.default_value
        return False

    def is_valid(self, row, column, value):
        return self.check_adjacent_diff_2(row, column, value) and self.check_biases_diff_1(row, column, value)

    # in every condition we check if it is already assigned, if not condition is satisfied
    def check_adjacent_diff_2(self, row, column, value):
        is_valid = True
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

    def check_biases_diff_1(self, row, column, value):
        is_valid = True
        df = self.default_value
        # upper left
        if column - 1 >= 0 and row - 1 >= 0:
            is_valid = self.matrix[row - 1, column - 1] == df or np.absolute(
                self.matrix[row - 1, column - 1] - value) >= 1
        # upper right
        if is_valid and column + 1 < self.matrix.shape[1] and row - 1 >= 0:
            is_valid = self.matrix[row - 1, column + 1] == df or np.absolute(
                self.matrix[row - 1, column + 1] - value) >= 1
        # lower left
        if is_valid and column - 1 >= 0 and row + 1 < self.matrix.shape[0]:
            is_valid = self.matrix[row + 1, column - 1] == df or np.absolute(
                self.matrix[row + 1, column - 1] - value) >= 1
        # lower bottom
        if is_valid and column + 1 < self.matrix.shape[1] and row + 1 < self.matrix.shape[0]:
            is_valid = self.matrix[row + 1, column + 1] == df or np.absolute(
                self.matrix[row + 1, column + 1] - value) >= 1
        return is_valid
