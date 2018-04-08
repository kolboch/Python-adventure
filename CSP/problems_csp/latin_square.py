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
        self.matrix = np.full(shape=(self.n, self.n), fill_value=self.default_value, dtype=int)
        solved = False
        while not solved:
            self.init_domains()
            solved = self.forward_solve_util(0, 0)
        return self.matrix

    def forward_solve_util(self, x, y):
        for value in self.domains_matrix[x][y]:
            self.matrix[x, y] = value
            self.validate_domains_matrix(x, y, value)
            if (x == self.n - 1 and y == self.n - 1) or self.forward_solve_util(*(self.next_indices(x, y))):
                return True
            self.matrix[x, y] = self.default_value
            self.restore_domains_from(x, y)
        return False

    def init_domains(self):
        self.domains_matrix = [[[z for z in range(1, self.n + 1)] for _ in range(self.n)] for _ in
                               range(self.n)]

    # EAFP rule here, if element already not in list just ignore exception
    def validate_domains_matrix(self, x, y, value):
        # all to the right
        for i in range(y, self.n):
            try:
                self.domains_matrix[x][i].remove(value)
            except ValueError:
                pass
        # all to the bottom
        for j in range(x, self.n):
            try:
                self.domains_matrix[j][y].remove(value)
            except ValueError:
                pass

    def restore_domains_from(self, x, y):
        # all to the right
        row_values = self.matrix[x]
        for i in range(y + 1, self.n):
            self.domains_matrix[x][i] = [z for z in range(1, self.n + 1) if
                                         z not in row_values and z not in [row[i] for row in self.matrix]]
        # all to the bottom
        column_values = [row[y] for row in self.matrix]
        for j in range(x + 1, self.n):
            self.domains_matrix[j][y] = [z for z in range(1, self.n + 1) if
                                         z not in column_values]

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
