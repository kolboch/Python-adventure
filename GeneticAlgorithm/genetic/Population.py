import numpy as np
import random
from random import shuffle
from crossovers.OrderCrossover import CrossoverOrdered
from genetic.Mutators import SwapMutator
from selection_algorithms.Selection import RouletteSelector
from qap.FitnessRater import ReversedFlowDistance


# argument for crossover selection ? / random/ do we need crossover probability?
class Population:
    def __init__(self, population_size=10, survive_percentage=70,
                 mutation_probability=0.08, best_survives=False,
                 crossover=CrossoverOrdered(), mutator=SwapMutator(),
                 selector=RouletteSelector(), fitness_rater=ReversedFlowDistance()):
        self._individuals = None
        self._population_size = population_size
        self._survive_percentage = survive_percentage
        self._mutation_probability = mutation_probability
        self._best_survives = best_survives
        self.crossover = crossover
        self.mutator = mutator
        self.selector = selector
        self.fitness_rater = fitness_rater

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

    def pass_generation(self):

        # fitness and selection
        scores = self.fitness_rater.compute_scores(self._individuals)
        to_choose = int(self._population_size * self._survive_percentage / 100)
        selected_indices = self.selector.select(scores, to_choose)
        selected_individuals = self._individuals[selected_indices]

        # crossover
        while selected_individuals.shape[0] < self._population_size:
            a, b = random.sample(range(to_choose), 2)
            ind_1, ind_2 = selected_individuals[a], selected_individuals[b]
            child_1, child_2 = self.crossover.crossover(ind_1, ind_2)
            selected_individuals = np.vstack((selected_individuals, np.vstack((child_1, child_2))))

        # mutation
        number_of_mutations = int(self._mutation_probability * self._population_size)
        for i in range(number_of_mutations):
            to_mutate = random.randint(0, self._population_size - 1)
            selected_individuals[to_mutate] = self.mutator.mutate(selected_individuals[to_mutate])

        self._individuals = selected_individuals
        return scores
