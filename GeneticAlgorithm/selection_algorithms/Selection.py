import random
from abc import ABC, abstractmethod

import numpy as np


class SelectorInterface(ABC):
    @abstractmethod
    def select(self, scores, choose_number):
        """
        :param scores: scores of individuals
        :param choose_number: number of individuals to choose
        :return: indices of chosen individuals
        """

    @staticmethod
    def _map_indices_to_scores(scores):
        """
            this method is meant to be private
        :param scores: scores of individuals, like [100, 90, 70]
        :return: indices mapped to scores, ex. {0: 100, 1: 90, 2: 70}
        """
        return dict(zip(range(scores.size), scores))


class RouletteSelector(SelectorInterface):
    def select(self, scores, choose_number):
        results = np.empty(choose_number, dtype=int)
        sum_of_scores = np.sum(scores)
        relative_score = scores / sum_of_scores
        n = len(relative_score)
        roulette_ranges = [np.sum(relative_score[:i + 1]) for i in range(n)]
        for i in range(choose_number):
            random_pick = np.random.random_sample()
            for j in range(n):
                if random_pick < roulette_ranges[j]:
                    results[i] = j
                    break
        return results


class TournamentSelector(SelectorInterface):
    """
        selects best individual from randomly selected group of size @group_size
    """

    def __init__(self, group_size=5):
        """
            :param group_size: size of tournament group, from which best individual is selected
        """
        self._group_size = group_size

    def select(self, scores, choose_count):
        indices_scores_dict = SelectorInterface._map_indices_to_scores(scores)
        print(indices_scores_dict)
        results = np.empty(choose_count, dtype=int)

        group_size = self._group_size
        for i in range(choose_count):
            items_left = len(indices_scores_dict)
            if items_left < self._group_size:
                group_size = items_left
            selected = random.sample(range(items_left), group_size)
            keys = [list(indices_scores_dict.keys())[i] for i in selected]
            values = [indices_scores_dict[key] for key in keys]
            min_value_index = np.argmin(values)
            min_item_index = keys[min_value_index]
            results[i] = min_item_index
            del indices_scores_dict[min_item_index]
        return results
