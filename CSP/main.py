from problems_csp.problems_CSP import GraphColorL21
from CSP import AlgorithmCSP

coloring = GraphColorL21(3, AlgorithmCSP.BACKTRACKING, None)
coloring.solve()
