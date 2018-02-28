from problem_generator.Generator import Generator

generator = Generator(problem_size=10, max_distance=40, max_flow_cost=20, symmetrical_flows=False)
distances, flows = generator.generate_problem()

print('distances \n {}'.format(distances))
print('flows \n {}'.format(flows))


# if __name__ == '__main__':
#     print("project init")
