from calculators.distance.base_distance_calculator import BaseDistanceCalculator


class PriceDistanceCalculator(BaseDistanceCalculator):

    @classmethod
    def calculate_distance(cls, input_price, compared_price, sensitivity=950):
        """
        Scores the price difference, normalized to a range of 0 to 1. If the compared price is lower, it scores 100.

        :param input_price: The reference price to compare against.
        :param compared_price: The price being compared.
        :param sensitivity: Adjusts the penalty for price differences when the compared price is higher.
        :return: The normalized score for the price difference.
        """
        if input_price == 0:  # Prevent division by zero
            return 1  # Assuming a score of 1 (100%) when there's no input price to compare against.

        price_diff = compared_price - input_price

        # If the compared price is lower, return a score close to 100
        if price_diff < 0:
            return 1  # Representing 99%, which is close to 100

        # For higher prices, increase the penalty
        score = max(0, 100 - (price_diff / input_price * sensitivity))

        return score / 100