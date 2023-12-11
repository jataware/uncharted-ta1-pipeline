import pickle
from collections import defaultdict


def cache_geocode_data(geocode_data, cache_path):
    with open(cache_path, "wb") as f:
        pickle.dump(geocode_data, f)


def load_geocode_cache(cache_path):
    geocode_data = defaultdict(list)
    try:
        with open(cache_path, "rb") as f:
            geocode_data = pickle.load(f)
    except Exception as e:
        pass
    finally:
        return geocode_data
