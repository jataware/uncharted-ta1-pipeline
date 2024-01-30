from pydantic import BaseModel


class PointOrientationConfig(BaseModel):
    """
    Configuration options for point symbol extraction
    (can be used to set point class-specific configurations)
    """

    point_class: str  # point symbol class label

    # do extraction of dip angle (2-digit label beside symbol)
    dip_number_extraction: bool = True

    template_rotate_interval: int = 5  # [deg] rotational interval for template matching
    bbox_size: int = 75  # [pixels] bounding-box size for symbol orientation analysis

    # [pixels] template cross-correlation search range (around point center)
    # (lower improves precision, higher improves recall)
    xcorr_search_range: int = 6

    # additional step to correct 180-deg "mirroring confusion",
    # (uses relative location of dip text label)
    mirroring_correction: bool = False
