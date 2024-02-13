import math


class GeoLocationParser:


    @classmethod
    def _haversine(cls, lat1, lon1, lat2, lon2):
        R = 6371  # Радиус на Земята в километри
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c
        return distance


    @classmethod
    def parse(cls, coordinates, threshold=2.0) -> dict:
        neighbors = {area: [] for area in coordinates}

        # Изчислете разстоянията и групирайте близките квартали
        for area1, (lat1, lon1) in coordinates.items():
            for area2, (lat2, lon2) in coordinates.items():
                if area1 != area2:
                    distance = cls._haversine(lat1, lon1, lat2, lon2)
                    if distance <= threshold:
                        neighbors[area1].append((area2, distance))
        areas = {}

        for area, close_neighbors in neighbors.items():
            area = area.replace(" ", "").lower()
            print(f"{area} е близо до:")
            areas[area] = [{'съсед': area, 'разстояние': 0}]
            for neighbor, distance in close_neighbors:
                areas[area].append(
                    {'съсед': neighbor.replace(" ", '').lower(), 'разстояние': float("{:.2f}".format(distance))})

        return areas
