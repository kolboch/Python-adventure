import random

from crossovers.CrossoverInterface import Crossover


class CrossoverOrdered(Crossover):
    """
        ordered crossover ---> OX operator,
        source before adjustment and code refactor - https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    """

    def crossover(self, individual_1, individual_2):
        size = min(len(individual_1), len(individual_2))
        left, right = random.sample(range(size), 2)
        if left > right:
            left, right = right, left

        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < left or i > right:  # omit checking range for cross
                holes1[individual_2[i]] = False
                holes2[individual_1[i]] = False

        # reorder except values which will come ( holes True not rewritten)
        temp1, temp2 = individual_1, individual_2
        start_point = right + 1
        k1, k2 = start_point, start_point
        for i in range(size):
            if not holes1[temp1[(i + start_point) % size]]:
                individual_1[k1 % size] = temp1[(i + start_point) % size]
                k1 += 1

            if not holes2[temp2[(i + start_point) % size]]:
                individual_2[k2 % size] = temp2[(i + start_point) % size]
                k2 += 1

        # Swap the content between left and right (included)
        for i in range(left, start_point):
            individual_1[i], individual_2[i] = individual_2[i], individual_1[i]

        return individual_1, individual_2
