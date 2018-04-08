from problems_csp.graph_coloring import GraphColorL21
from CSP import AlgorithmCSP
from matplotlib import colors
import matplotlib.pyplot as plt
from problems_csp.latin_square import LatinSquare
import sys

# sys.setrecursionlimit(2000)
# problem_size = 13
# cmap = colors.ListedColormap(['lightblue', 'lightgreen', 'yellow', 'orange', 'grey'])
#
# coloring = GraphColorL21(problem_size, AlgorithmCSP.BACKTRACKING, None)
# result = coloring.solve()
# print(result)
#
# img = plt.imshow(result, interpolation='nearest', origin='lower', cmap=cmap)
# plt.colorbar()
# plt.title("Graph colouring L(2, 1)")
# plt.show()

latin_problem_size = 10
cmap_latin = plt.cm.jet
latin = LatinSquare(latin_problem_size, AlgorithmCSP.FORWARD_CHECKING, None)
result_latin = latin.solve()

img = plt.imshow(result_latin, interpolation='nearest', origin='lower', cmap=cmap_latin)
plt.grid = False
plt.colorbar()
plt.title("Latin square, N= {}".format(latin_problem_size))
plt.show()

print(result_latin)


