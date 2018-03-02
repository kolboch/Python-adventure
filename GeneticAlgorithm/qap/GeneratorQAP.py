import numpy as np


class GeneratorQAP:
    def __init__(self, problem_size=5, min_distance=1, max_distance=20, min_flow_cost=0, max_flow_cost=20,
                 symmetrical_flows=True):
        self._problem_size = problem_size
        self._symmetrical_flows = symmetrical_flows

        self._minimum_distance = min_distance if min_distance > 0 else 1
        self._maximum_distance = max_distance if max_distance > self._minimum_distance else self._minimum_distance + 1

        self._min_flow_cost = min_flow_cost if min_flow_cost >= 0 else 0
        self._max_flow_cost = max_flow_cost if max_flow_cost > self._min_flow_cost else self._min_flow_cost + 1

        self._distances = None
        self._flows = None

    @property
    def problem_size(self):
        return self._problem_size

    @problem_size.setter
    def problem_size(self, size):
        if size > 0:
            self._problem_size = size
        else:
            raise ValueError('Problem size must be greater than 0')

    @property
    def maximum_distance(self):
        return self._maximum_distance

    @maximum_distance.setter
    def maximum_distance(self, max_dist):
        if max_dist > 0:
            self._maximum_distance = max_dist
        else:
            raise ValueError('Max distance must be greater than 0')

    @property
    def max_flow_cost(self):
        return self._max_flow_cost

    @max_flow_cost.setter
    def max_flow_cost(self, max_flow):
        if max_flow > 0:
            self._max_flow_cost = max_flow
        else:
            raise ValueError('Max flow cost must be greater than 0')

    def generate_distances(self):
        random_x = np.random.randint(self._minimum_distance, self._maximum_distance,
                                     (self._problem_size, self._problem_size))
        self._distances = (random_x + random_x.T) / 2
        np.fill_diagonal(self._distances, 0)
        return self._distances

    def generate_flows(self):
        random_x = np.random.randint(self._min_flow_cost, self._max_flow_cost,
                                     (self._problem_size, self._problem_size))
        if self._symmetrical_flows:
            self._flows = (random_x + random_x.T) / 2
        else:
            self._flows = random_x
        np.fill_diagonal(self._flows, 0)
        return self._flows

    def generate_problem(self):
        return self.generate_distances(), self.generate_flows()
