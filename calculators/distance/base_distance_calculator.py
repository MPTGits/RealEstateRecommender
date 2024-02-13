from abc import ABC, abstractmethod


class BaseDistanceCalculator(ABC):

    @abstractmethod
    def calculate_distance(self, *args, **kwargs):
        pass