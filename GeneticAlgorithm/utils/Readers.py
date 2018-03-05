import numpy as np


class FileReader:
    @staticmethod
    def read_problem_file(file_name):
        """
            file must be in format:
        :param file_name:
        :return: distances, flows, problem_size
        """
        with open(file_name, mode='r') as f:
            first_line = f.readline()
        dist_flow_data = np.loadtxt(file_name, skiprows=1, dtype=int)
        problem_size = int(first_line)
        distances = dist_flow_data[:problem_size]
        flows = dist_flow_data[problem_size:]
        return problem_size, distances, flows

    @staticmethod
    def read_solution(file_name):
        solution = np.loadtxt(file_name, dtype=int)
        return solution
