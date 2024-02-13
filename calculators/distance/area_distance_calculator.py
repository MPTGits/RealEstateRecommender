from calculators.distance.base_distance_calculator import BaseDistanceCalculator


class AreaDistanceCalculator(BaseDistanceCalculator):

    @classmethod
    def calculate_distance(cls, input_area, compared_area, sensitivity=100):
        """
          Scores the area difference, normalized to a range of 0 to 1.

          :param input_area: The reference area to compare against.
          :param compared_area: The area being compared.
          :param sensitivity: Adjusts the penalty for area differences.
          :return: The normalized score for the area difference.
          """
        if input_area == 0:  # Prevent division by zero
            return 0

        area_diff = abs(compared_area - input_area) / input_area
        score = max(0, 100 - (area_diff * sensitivity))

        return score / 100