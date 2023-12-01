import numpy as np

from geopy.distance import distance as geo_distance
from sklearn.cluster import DBSCAN

FOV_RANGE_KM = 700


def split_lon_lat_degrees(
    geofence, degrees: list[float, float, float]
) -> (list[float], list[float]):
    if len(degrees) == 0:
        return geofence

    # start with clustering approach
    # only need a few to cluster to determine lat and lon range
    data = np.array(list(map(lambda x: x[2], degrees))).reshape(-1, 1)
    print(f"clustering data: {data}")

    db = DBSCAN(eps=0.3, min_samples=2).fit(data)
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)
    print(f"labels: {labels}")

    # determine which of the clusters could be lat vs lon
    data_clustered = list(zip(degrees, labels))
    print(f"cluster results: {data_clustered}")

    clusters = {}
    for d in data_clustered:
        if d[1] != -1:
            if d[1] not in clusters:
                clusters[d[1]] = []
            clusters[d[1]].append(d[0])

    # assume lat & lon will be biggest 2 clusters
    cluster_list = []
    for _, c in clusters.items():
        cluster_list.append(c)

    # handle the case where fewer than 2 clusters are detected
    print(f"initial geofence: {geofence}")
    if len(cluster_list) < 1:
        return geofence
    elif len(cluster_list) == 1:
        ok, is_lat = cluster_is_lat(geofence, cluster_list[0])
        if ok:
            degrees = list(map(lambda x: x[2], cluster_list[0]))
            updated_geofence = (geofence[0], geofence[1])
            if is_lat:
                print("lat determined")
                updated_lat = narrow_geofence(
                    (min(degrees), max(degrees)), geofence[0], FOV_RANGE_KM
                )
                updated_geofence = (geofence[0], updated_lat[1])
            else:
                print("lon determined")
                updated_lon = narrow_geofence(
                    geofence[1], (min(degrees), max(degrees)), FOV_RANGE_KM
                )
                updated_geofence = (updated_lon[0], geofence[1])
            return updated_geofence
        else:
            return geofence

    cluster_list = sorted(cluster_list, key=lambda x: len(x), reverse=True)

    c1 = cluster_list[0]
    c2 = cluster_list[1]
    degrees_c1 = list(map(lambda x: x[2], c1))
    degrees_c2 = list(map(lambda x: x[2], c2))

    lat, lon = None, None
    ok, is_lat = cluster_is_lat(geofence, c1)
    if ok:
        if is_lat:
            lat = (min(degrees_c1), max(degrees_c1))
            lon = (min(degrees_c2), max(degrees_c2))
        else:
            lat_cluster = find_lat_cluster(geofence, cluster_list[1:])
            if lat_cluster:
                degrees_c2 = list(map(lambda x: x[2], lat_cluster))
            lat = (min(degrees_c2), max(degrees_c2))
            lon = (min(degrees_c1), max(degrees_c1))
    else:
        ok, is_lat = cluster_is_lat(geofence, c2)
        if ok:
            if is_lat:
                lat = (min(degrees_c2), max(degrees_c2))
                lon = (min(degrees_c1), max(degrees_c1))
            else:
                lat = (min(degrees_c1), max(degrees_c1))
                lon = (min(degrees_c2), max(degrees_c2))

    if lat and lon:
        # make sure lat is under lat limit of 90 otherwise only update lon
        if lat[0] < 90:
            return narrow_geofence(lat, lon, FOV_RANGE_KM)
        else:
            updated_lon = narrow_geofence(geofence[1], lon, FOV_RANGE_KM)
            return (updated_lon[0], geofence[1])
    return geofence


def cluster_is_lat(geofence, degrees: list[float, float, float]) -> (bool, bool):
    print(f"lat check: {degrees}")
    # latitude cannot be over 90
    if abs(degrees[0][2]) > 90:
        return True, False

    # value could fall in one geofence while being outside the other
    in_lat_geofence = is_in_geofence(geofence[1], degrees[0][2])
    in_lon_geofence = is_in_geofence(geofence[0], degrees[0][2])
    if in_lat_geofence and not in_lon_geofence:
        return True, True
    elif not in_lat_geofence and in_lon_geofence:
        return True, False

    # determine if x or y changes within cluster values

    # unable to determine direction
    return False, False


def is_in_geofence(geofence, degree: float) -> bool:
    return geofence[0] <= degree <= geofence[1]


def narrow_geofence(lat: (float, float), lon: (float, float), fov_range_km: float):
    dist_km = fov_range_km / 2.0  # distance from clue pt in all directions (N,E,S,W)
    fov_pt_north = geo_distance(kilometers=dist_km).destination(
        (lat[0], lon[0]), bearing=0
    )
    fov_pt_east = geo_distance(kilometers=dist_km).destination(
        (lat[0], lon[0]), bearing=90
    )
    fov_degrange_lon = abs(fov_pt_east[1] - lon[0])
    fov_degrange_lat = abs(fov_pt_north[0] - lat[0])
    lon_minmax = [lon[0] - fov_degrange_lon, lon[1] + fov_degrange_lon]
    lat_minmax = [lat[0] - fov_degrange_lat, lat[1] + fov_degrange_lat]

    return (lon_minmax, lat_minmax)


def find_lat_cluster(geofence, clusters):
    for c in clusters:
        ok, is_lat = cluster_is_lat(geofence, c)
        if ok and is_lat:
            return c
    return None
