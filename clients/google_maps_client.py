import requests

class GoogleMapsClient:

    def __init__(self, api_key):
        self.api_key = api_key

    @classmethod
    def get_coordinates(self, address, api_key):
        """Връща географските координати на даден адрес."""
        url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                location = data['results'][0]['geometry']['location']
                return (location['lat'], location['lng'])
        return None