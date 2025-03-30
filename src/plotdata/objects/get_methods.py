from abc import ABC, abstractmethod
import requests


class DataFetcher(ABC):
    def __init__(self, base_url, params):
        self.base_url = base_url
        self.params = params

    @abstractmethod
    def construct_url(self, *args, **kwargs):
        pass

    def fetch_data(self, *args, **kwargs):
        url = self.construct_url(*args, **kwargs)
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


class DataFetcherFactory:
    @staticmethod
    def create_fetcher(website, **kwargs):
        if website == "usgs":
            print("#" * 50)
            print("USGS database")
            print("#" * 50)
            print()

            return USGSDataFetcher(**kwargs)
        elif website == "anotherwebsite":
            return AnotherWebsiteDataFetcher(**kwargs)
        else:
            raise ValueError(f"Unknown website: {website}")


class USGSDataFetcher(DataFetcher):
    API_ENDPOINT = "https://earthquake.usgs.gov/fdsnws/event/1/query"

    def __init__(self, start_date, end_date, magnitude, params=None):
        super().__init__(self.API_ENDPOINT, params or {})
        self.start_date = start_date
        self.end_date = end_date
        self.magnitude = magnitude

    def construct_url(self, max_lat, min_lat, max_lon, min_lon):
        self.params.update({
            "format": "geojson",
            "starttime": self.start_date,
            "endtime": self.end_date,
            "maxlatitude": max_lat,
            "minlatitude": min_lat,
            "maxlongitude": max_lon,
            "minlongitude": min_lon,
            "minmagnitude": self.magnitude
        })
        return f"{self.base_url}?{'&'.join([f'{k}={v}' for k, v in self.params.items()])}"


class AnotherWebsiteDataFetcher(DataFetcher):
    API_ENDPOINT = "https://anotherwebsite.com/api/data"

    def __init__(self, some_param, params=None):
        super().__init__(self.API_ENDPOINT, params or {})
        self.some_param = some_param

    def construct_url(self):
        self.params.update({
            "param": self.some_param
        })
        return f"{self.base_url}?{'&'.join([f'{k}={v}' for k, v in self.params.items()])}"