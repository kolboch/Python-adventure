import numpy as np
from qap.GeneratorQAP import GeneratorQAP
from genetic.Population import Population
from qap.FitnessRater import ReversedFlowDistance
from selection_algorithms.Selection import TournamentSelector, RouletteSelector
from crossovers.OrderCrossover import CrossoverOrdered
from crossovers.PartialMappedCrossover import CrossoverPartiallyMapped
from crossovers.CycleCrossover import CrossoverCycle

problem_size = 12
tournament_group_size = 5
survivability_percentage = 75
population_size = 1000
mutation_probability = 0.2
generations_to_simulate = 200

generator = GeneratorQAP(problem_size=problem_size, max_distance=40, max_flow_cost=5, symmetrical_flows=True)
distances, flows = generator.generate_problem()
score_rater = ReversedFlowDistance(distances, flows)
# selector = TournamentSelector(tournament_group_size)
population = Population(population_size=population_size, survive_percentage=survivability_percentage,
                        mutation_probability=mutation_probability, fitness_rater=score_rater)
population.generate(problem_size)
generations_scores = np.zeros(generations_to_simulate)
for gen in range(generations_to_simulate):
    if gen % 10 == 0:
        print('starting generation: {}'.format(gen))
    generations_scores[gen] = np.sum(population.pass_generation())

print('{}'.format(generations_scores))
