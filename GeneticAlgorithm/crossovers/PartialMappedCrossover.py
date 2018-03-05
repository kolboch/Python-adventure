import random

from crossovers.CrossoverInterface import Crossover


class CrossoverPartiallyMapped(Crossover):
    """
        partial mapped crossover ---> PMX operator
        source before adjustment and code refactor - https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    """

    def crossover(self, individual_1, individual_2):
        size = min(len(individual_1), len(individual_2))
        p1, p2 = [0] * size, [0] * size
        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[individual_1[i]] = i
            p2[individual_2[i]] = i
        # now p1, p2 are answering questions ex. where do I find value 6 in individual_1 -> at p1[6]
        # Choose crossover points
        left, right = random.sample(range(size), 2)
        if left > right:
            left, right = right, left

        # Apply crossover between cx points
        for i in range(left, right):
            # Keep track of the selected values
            temp1 = individual_1[i]
            temp2 = individual_2[i]
            # Swap the matched value
            individual_1[i], individual_1[p1[temp2]] = temp2, temp1
            individual_2[i], individual_2[p2[temp1]] = temp1, temp2
            # Position bookkeeping
            p1[temp1], p1[temp2] = p1[temp2], p1[temp1]
            p2[temp1], p2[temp2] = p2[temp2], p2[temp1]

        return individual_1, individual_2
