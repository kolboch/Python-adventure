from qap.GeneratorQAP import GeneratorQAP
from genetic.Population import Population
from qap.FitnessRater import ReversedFlowDistance

problem_size = 10

generator = GeneratorQAP(problem_size=problem_size, max_distance=40, max_flow_cost=20, symmetrical_flows=False)
distances, flows = generator.generate_problem()
score_rater = ReversedFlowDistance(distances, flows)

population = Population(population_size=5)
population.generate(problem_size)
individuals = population.individuals
print(individuals.shape[0])

scores = score_rater.compute_scores(individuals)

print('distances \n {}'.format(distances))
print('flows \n {}'.format(flows))
print('individuals \n {}'.format(individuals))
print('scores \n {}'.format(scores))
