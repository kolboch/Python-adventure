from CSP import AlgorithmCSP
from problems_csp.problems_CSP import ProblemCSP
import numpy as np


# domains {1,2, ... 9 }
# constraints: in every row and every column given value only once
#
class LatinSquare(ProblemCSP):
    def __init__(self, problem_size, algorithm, heuristic):
        super().__init__(problem_size, algorithm, heuristic)
        self.domain = np.arange(1, problem_size + 1)
        self.default_value = 0

    def solve(self):
        if self.algorithm == AlgorithmCSP.FORWARD_CHECKING:
            return self.solve_forward_checking()
        elif self.algorithm == AlgorithmCSP.BACKTRACKING:
            return self.solve_backtracking()
        else:
            raise ValueError('This value is not supported. Choose one of AlgorithmCSP values.')

    def solve_forward_checking(self):
        pass

    def solve_backtracking(self):
        self.matrix = np.full(shape=(self.n, self.n), fill_value=0, dtype=int)
        solved = False
        while not solved:
            solved = self.solve_util(0, 0)
        return self.matrix

    def solve_util(self, x, y):
        for value in self.domain:
            if self.is_valid(x, y, value):
                self.matrix[x, y] = value
                if (x == self.n - 1 and y == self.n - 1) or self.solve_util(*(self.next_indices(x, y))):
                    return True
                self.matrix[x, y] = self.default_value
        return False

    def is_valid(self, x, y, value):
        return self.is_column_valid(y, value) and self.is_row_valid(x, value)

    def is_column_valid(self, y, value):
        return value not in self.matrix[:, y]

    def is_row_valid(self, x, value):
        return value not in self.matrix[x, :]
