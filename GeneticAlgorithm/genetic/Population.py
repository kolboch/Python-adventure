import numpy as np
from random import shuffle
from crossovers.OrderCrossover import CrossoverOrdered


# argument for crossover selection ? / random/ do we need crossover probability?
class Population:
    def __init__(self, population_size=10, survive_percentage=70,
                 mutation_probability=0.08, best_survives=False, crossover=CrossoverOrdered()):
        self._individuals = None
        self._population_size = population_size
        self._survive_percentage = survive_percentage
        self._mutation_probability = mutation_probability
        self._best_survives = best_survives
        self._crossover = crossover

    @property
    def individuals(self):
        return self._individuals

    @individuals.setter
    def individuals(self, individuals):
        self._individuals = individuals

    def generate(self, problem_size):
        pop_size = self._population_size
        individuals = np.empty(shape=(pop_size, problem_size), dtype=int)
        for i in range(pop_size):
            temp = [i for i in range(problem_size)]
            shuffle(temp)
            individuals[i, :] = temp
        self._individuals = individuals

    def pass_generation(self, scores):
        # eliminate poor performing ones aka selection
        # generate children ----- crossover
        # mutate random guys ;) ----- mutation method
        pass
