from qap.GeneratorQAP import GeneratorQAP
from genetic.Population import Population

problem_size = 10

generator = GeneratorQAP(problem_size=problem_size, max_distance=40, max_flow_cost=20, symmetrical_flows=False)
distances, flows = generator.generate_problem()

population = Population(population_size=5)
population.generate(problem_size)
individuals = population.individuals


print('distances \n {}'.format(distances))
print('flows \n {}'.format(flows))
print('individuals \n {}'.format(individuals))
