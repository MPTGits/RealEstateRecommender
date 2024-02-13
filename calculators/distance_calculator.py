from calculators.distance.base_distance_calculator import BaseDistanceCalculator


class DistanceCalculator:
    def __init__(self, calculator: BaseDistanceCalculator=None):
        self._calculator = calculator

    def calculate(self, *args, **kwargs):
        return self._calculator.calculate_distance(*args, **kwargs)

    def set_calculator(self, calculator: BaseDistanceCalculator):
        self._calculator = calculator