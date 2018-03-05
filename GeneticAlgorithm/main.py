import numpy as np
from qap.GeneratorQAP import GeneratorQAP
from genetic.Population import Population
from qap.FitnessRater import ReversedFlowDistance
from selection_algorithms.Selection import TournamentSelector, RouletteSelector
from crossovers.OrderCrossover import CrossoverOrdered
from crossovers.PartialMappedCrossover import CrossoverPartiallyMapped
from crossovers.CycleCrossover import CrossoverCycle
from genetic.Mutators import SwapMutator, InvertMutator, ShuffleMutator
from utils import Readers
import matplotlib.pyplot as plt

problem_size = 18
tournament_group_size = 5
survivability_percentage = 70
population_size = 200
mutation_probability = 0.08
generations_to_simulate = 100

file_problem_size = "16"

problem_file_path = f"problems/had{file_problem_size}.dat"
solution_file_path = f"problems/solution{file_problem_size}.txt"
solution = Readers.FileReader.read_solution(solution_file_path)
solution -= 1
solution = np.array([solution])
problem_size, distances, flows = Readers.FileReader.read_problem_file(problem_file_path)

generator = GeneratorQAP(problem_size=problem_size, max_distance=40, max_flow_cost=5, symmetrical_flows=True)

score_rater = ReversedFlowDistance(distances, flows)
solution_score = score_rater.compute_scores(solution)
# selector = TournamentSelector(tournament_group_size)
population = Population(population_size=population_size, survive_percentage=survivability_percentage,
                        mutation_probability=mutation_probability, fitness_rater=score_rater,
                        crossover=CrossoverCycle(), mutator=ShuffleMutator())

simulations_todo = 2
simulation_results = np.empty(shape=(simulations_todo, 4, generations_to_simulate))
for i in range(simulations_todo):
    population.generate(problem_size)
    generations_scores = np.empty(shape=(generations_to_simulate, population_size))

    for gen in range(generations_to_simulate):
        generations_scores[gen] = population.pass_generation()

    maxes, averages, mins, stds = np.amax(generations_scores, axis=1), np.mean(generations_scores, axis=1), np.amin(
        generations_scores, axis=1), np.std(generations_scores, axis=1)
    simulation_results[i] = [maxes, averages, mins, stds]

# print(simulation_results)
# test = simulation_results[:, 0, :]
# test = np.mean(test.T, axis=1)
#
# print('test \n{}'.format(test))

maxes, averages, mins, stds = np.mean(simulation_results[:, 0, :].T, axis=1), \
                              np.mean(simulation_results[:, 1, :].T, axis=1),\
                              np.mean(simulation_results[:, 2, :].T, axis=1), \
                              np.mean(simulation_results[:, 3, :].T, axis=1)


solution_line = plt.axhline(solution_score, label="Solution score", linewidth=1, color='r')
maxes_line, = plt.plot(maxes, label="Maximum", linewidth=1)
avg_line, = plt.plot(averages, label="Mean", linewidth=1)
min_line, = plt.plot(mins, label="Minimum", linewidth=1)
plt.legend(handles=[maxes_line, avg_line, min_line, solution_line], loc=1)
plt.title('Population performance')
plt.ylabel('Score')
plt.xlim(xmin=0)
plt.xlabel('generation number')
plt.show()
