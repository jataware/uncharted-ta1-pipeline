
def ocr_to_coordinates(ocr_bounding_poly):
    mapped = []
    for v in ocr_bounding_poly.vertices:
        mapped.append([v.x, v.y])
    return mapped