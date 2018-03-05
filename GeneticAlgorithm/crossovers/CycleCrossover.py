import numpy as np

from crossovers.CrossoverInterface import Crossover


class CrossoverCycle(Crossover):
    """
        cycle crossover ---> CX operator
    """

    def crossover(self, individual_1, individual_2):
        size = min(len(individual_1), len(individual_2))
        p1, p2 = [0] * size, [0] * size
        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[individual_1[i]] = i
            p2[individual_2[i]] = i

        # finding cycle
        first_elem = individual_1[0]
        cycle_elements = [first_elem]
        current = first_elem
        for i in range(size):
            respondent = individual_2[p1[current]]
            current = individual_1[p1[respondent]]
            if current != first_elem:
                cycle_elements.append(current)
            else:
                # we found the cycle
                break

        # creating children
        result_1, result_2 = np.array([None] * size), np.array([None] * size)
        for val in cycle_elements:
            result_1[p1[val]] = val
            result_2[p2[val]] = val

        for i in range(size):
            if result_1[i] is None:
                result_1[i] = individual_2[i]
            if result_2[i] is None:
                result_2[i] = individual_1[i]
        return result_1, result_2
