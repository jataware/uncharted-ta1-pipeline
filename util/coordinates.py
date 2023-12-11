def ocr_to_coordinates(ocr_bounding_poly):
    mapped = []
    for v in ocr_bounding_poly.vertices:
        mapped.append([v.x, v.y])
    return mapped


def absolute_minmax(minmax: list[float]) -> list[float]:
    minmax_abs = minmax.copy()
    # if the min max crosses 0, need to have it span from 0 to the furthest value
    if minmax_abs[0] < 0 and minmax_abs[1] >= 0:
        minmax_abs = [0, max(abs(minmax_abs[0]), abs(minmax_abs[1]))]
    else:
        minmax_abs = [abs(minmax_abs[0]), abs(minmax_abs[1])]
        minmax_abs = [min(minmax_abs), max(minmax_abs)]
    return minmax_abs
