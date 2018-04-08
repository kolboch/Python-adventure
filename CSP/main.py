from problems_csp.problems_CSP import GraphColorL21
from CSP import AlgorithmCSP
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np

problem_size = 5
cmap = colors.ListedColormap(['lightblue', 'lightgreen', 'yellow', 'orange', 'grey'])

coloring = GraphColorL21(problem_size, AlgorithmCSP.BACKTRACKING, None)
result = coloring.solve()
print(result)

img = plt.imshow(result, interpolation='nearest', origin='lower', cmap=cmap)
plt.colorbar()
plt.title("Graph colouring L(2, 1)")
plt.show()
#
# colors = 3
# x = 5
# y = 5
# my_list = [[[z for z in range(colors)] for y in range(3)] for x in range(2)]
# del my_list[0][0][0]
# print(my_list)
