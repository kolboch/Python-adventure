from crossovers.CrossoverInterface import Crossover
import random


class CrossoverOrdered(Crossover):
    """
        ordered crossover ---> OX,
        source before adjustment - https://github.com/DEAP/deap/blob/master/deap/tools/crossover.py
    """

    def crossover(self, individual_1, individual_2):
        size = min(len(individual_1), len(individual_2))
        a, b = random.sample(range(size), 2)
        if a > b:
            a, b = b, a
        # TODO temp to be deleted
        a, b = 3, 5
        print('a: {} b: {}'.format(a, b))
        holes1, holes2 = [True] * size, [True] * size
        for i in range(size):
            if i < a or i > b:  # omit checking range for cross
                holes1[individual_2[i]] = False
                holes2[individual_1[i]] = False

        print('holes1: {}, \n holes2:{}'.format(holes1, holes2))
        temp1, temp2 = individual_1, individual_2
        k1, k2 = b + 1, b + 1
        for i in range(size):
            if not holes1[temp1[(i + b + 1) % size]]:
                individual_1[k1 % size] = temp1[(i + b + 1) % size]
                k1 += 1

            if not holes2[temp2[(i + b + 1) % size]]:
                individual_2[k2 % size] = temp2[(i + b + 1) % size]
                k2 += 1

        # Swap the content between a and b (included)
        for i in range(a, b + 1):
            individual_1[i], individual_2[i] = individual_2[i], individual_1[i]

        return individual_1, individual_2
