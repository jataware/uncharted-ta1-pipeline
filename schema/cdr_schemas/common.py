from enum import Enum


class GeomType(str, Enum):
    Point = "Point"
    LineString = "LineString"
    Polygon = "Polygon"


class GeoJsonType(str, Enum):
    Feature = "Feature"
    FeatureCollection = "FeatureCollection"
