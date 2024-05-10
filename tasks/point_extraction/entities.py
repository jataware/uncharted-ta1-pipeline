from __future__ import annotations

import logging
from PIL import Image
from pydantic import BaseModel, validator, Field
from typing import Optional, List, Union, Any, Dict

logger = logging.getLogger(__name__)
## Data Objects

LEGEND_ITEMS_OUTPUT_KEY = "legend_point_items"


class MapPointLabel(BaseModel):
    """
    Represents a label on a map image.
    Class ID should correspond to the ID encoded in the underlying model.
    """

    classifier_name: str
    classifier_version: str
    class_id: int
    class_name: str
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    direction: Optional[float] = None  # [deg] orientation of point symbol
    dip: Optional[float] = None  # [deg] dip angle associated with symbol
    legend_name: str
    legend_bbox: List[Union[float, int]]


class MapImage(BaseModel):
    """
    Represents a map image with point symbol prediction results
    """

    path: str
    raster_id: str
    labels: Optional[List[MapPointLabel]] = None
    map_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of map. TODO: Accept polygonal seg mask.
    )
    point_legend_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of point legend.
    )
    polygon_legend_bounds: Optional[List[int]] = (
        None  # [x1, y1, h, w] location of polygon legend.
    )

    _cached_image = None

    @property
    def size(self):
        # Make use of the PIL Image.open lazy image loading to avoid loading the image prematurely.
        return self.image.size

    @property
    def image(self):
        if self._cached_image:
            img = self._cached_image
        else:
            img = Image.open(self.path)
            if img.size[0] == 0 or img.size[1] == 0:
                raise ValueError("Image cannot have 0 height or width")
            self._cached_image = img
        # TODO: Use polygonal segmask stored in self.map_bounds to filter the image and crop out the non-map regions.
        return img


class MapTile(BaseModel):
    """
    Represents a tile of a map image in (x, y, width, height) format.
    x and y are coordinates on the original map image.

    Image tensors are assumed to be in Torchvision format (C, H, W). These are automatically converted to PIL Images.
    """

    x_offset: int  # x offset of the tile in the original image.
    y_offset: int  # y offset of the tile in the original image.
    width: int
    height: int
    map_bounds: tuple  # map global bounds (x_min, y_min, x_max, y_max)
    image: Any  # torch.Tensor or PIL.Image
    map_path: str  # Path to the original map image.
    predictions: Optional[List[MapPointLabel]] = None

    @validator("image", pre=True, always=True)
    def validate_image(cls, value):
        if value is None:
            return value
        if not isinstance(value, Image.Image):
            raise TypeError(f"Expected PIL or torch.Tensor Image, got {type(value)}")
        if value.size[0] == 0 or value.size[1] == 0:
            raise ValueError("Image cannot have 0 height or width")

        return value

    class Config:
        arbitrary_types_allowed = True


class MapTiles(BaseModel):
    raster_id: str
    tiles: List[MapTile]

    def format_for_caching(self) -> MapTiles:
        """
        Reformat point extraction tiles prior to caching
        - tile image raster is discarded
        """

        tiles_cache = []
        for t in self.tiles:
            t_cache = MapTile(
                x_offset=t.x_offset,
                y_offset=t.y_offset,
                width=t.width,
                height=t.height,
                map_bounds=t.map_bounds,
                image=None,
                map_path=t.map_path,
                predictions=t.predictions,
            )
            tiles_cache.append(t_cache)

        return MapTiles(raster_id=self.raster_id, tiles=tiles_cache)

    def join_with_cached_predictions(
        self, cached_preds: MapTiles, point_legend_mapping: Dict[str, LegendPointItem]
    ) -> bool:
        """
        Append cached point predictions to MapTiles
        """
        try:
            # re-format cached predictions with key as (x_offset, y_offset)
            cached_dict: Dict[Any, MapTile] = {}
            for p in cached_preds.tiles:
                cached_dict[(p.x_offset, p.y_offset)] = p
            for t in self.tiles:
                key = (t.x_offset, t.y_offset)
                if key not in cached_dict:
                    # cached predictions not found for this tile!
                    return False
                t_cached: MapTile = cached_dict[key]
                t.predictions = t_cached.predictions

                if t.predictions is not None:
                    for pred in t.predictions:
                        class_name = pred.class_name
                        # map YOLO class name to legend item name, if available
                        if pred.class_name in point_legend_mapping:
                            pred.legend_name = point_legend_mapping[class_name].name
                            pred.legend_bbox = point_legend_mapping[
                                class_name
                            ].legend_bbox

            return True
        except Exception as e:
            print(f"Exception in join_with_cached_predictions: {str(e)}")
            return False


class LegendPointItem(BaseModel):
    """
    Class for internally storing info about legend items for point symbols
    """

    # TODO -- could be modified to use CDR PointLegendAndFeaturesResult class in the future

    name: str = Field(description="Label of the map unit in the legend")
    description: str = Field(
        default="", description="Description of the map unit in the legend"
    )
    legend_bbox: List[Union[float, int]] = Field(
        default_factory=list,
        description="""The rough 2 point bounding box of the map units label.
                    Format is expected to be [x1,y1,x2,y2] where the top left
                    is the origin (0,0).""",
    )
    legend_contour: List[List[Union[float, int]]] = Field(
        default_factory=list,
        description="""The more precise polygon bounding box of the map units
                    label. Format is expected to be [x,y] coordinate pairs
                    where the top left is the origin (0,0).""",
    )


class LegendPointItems(BaseModel):
    """
    Class for a collection of LegendPointItem objects
    """

    items: List[LegendPointItem]
    provenance: str = Field(
        default="", description="where did these legend items come from"
    )
