from calculators.distance.base_distance_calculator import BaseDistanceCalculator


class GeoDistanceCalculator(BaseDistanceCalculator):
    # Default distance if no direct neighborhood relation is found
    DEFAULT_DISTANCE_OFFSET = 5.0  # Arbitrary high value to signify 'far'

    @classmethod
    def calculate_distance(cls, input_area, listing_area, neighborhood_mapping):
        if input_area in neighborhood_mapping:
            neighbors = neighborhood_mapping[input_area]
            for neighbor in neighbors:
                if neighbor['съсед'] == listing_area:
                    return neighbor['разстояние']
        return cls.DEFAULT_DISTANCE_OFFSET