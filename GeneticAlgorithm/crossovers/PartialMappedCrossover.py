from crossovers.CrossoverInterface import Crossover
import random


class CrossoverPartiallyMapped(Crossover):
    def crossover(self, individual_1, individual_2):
        size = min(len(individual_1), len(individual_2))
        p1, p2 = [0] * size, [0] * size

        # Initialize the position of each indices in the individuals
        for i in range(size):
            p1[individual_1[i]] = i
            p2[individual_2[i]] = i
        # Choose crossover points
        cxpoint1 = random.randint(0, size)
        cxpoint2 = random.randint(0, size - 1)
        if cxpoint2 >= cxpoint1:
            cxpoint2 += 1
        else:  # Swap the two cx points
            cxpoint1, cxpoint2 = cxpoint2, cxpoint1

        # Apply crossover between cx points
        for i in range(cxpoint1, cxpoint2):
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
