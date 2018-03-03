import numpy as np
from qap.GeneratorQAP import GeneratorQAP
from genetic.Population import Population
from qap.FitnessRater import ReversedFlowDistance
from selection_algorithms.Selection import TournamentSelector, RouletteSelector

sample_distances = np.array([[0, 22, 53, 53],
                             [22, 0, 40, 62],
                             [53, 40, 0, 55],
                             [53, 62, 55, 0]])
sample_flows = np.array([[0, 3, 0, 2],
                         [3, 0, 0, 1],
                         [0, 0, 0, 4],
                         [2, 1, 4, 0]])

problem_size = 10
tournament_group_size = 5
survivability_percentage = 10
population_size = 20
# sample sample sample

# score_rater_sample = ReversedFlowDistance(sample_distances, sample_flows)
#
# sample_individuals = np.array([[0, 1, 2, 3], [3, 1, 0, 2], [2, 3, 0, 1]])
# scores_sample = score_rater_sample.compute_scores(sample_individuals)
# print('distances \n {}'.format(sample_distances))
# print('flows \n {}'.format(sample_flows))
# print('individuals {}'.format(sample_individuals))
# print('scores {}'.format(scores_sample))

# end of sample

generator = GeneratorQAP(problem_size=problem_size, max_distance=40, max_flow_cost=20, symmetrical_flows=False)
distances, flows = generator.generate_problem()
score_rater = ReversedFlowDistance(distances, flows)
# selector = TournamentSelector(tournament_group_size)
selector = RouletteSelector()
population = Population(population_size=population_size, survive_percentage=survivability_percentage)
population.generate(problem_size)
individuals = population.individuals

scores = score_rater.compute_scores(individuals)
selector.select(scores, int(population_size * survivability_percentage / 100))

# print('distances \n {}'.format(distances))
# print('flows \n {}'.format(flows))
# print('individuals \n {}'.format(individuals))
# print('scores \n {}'.format(scores))
