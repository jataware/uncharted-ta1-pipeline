from tasks.point_extraction.entities import ImageTile
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Dict


class PointInferenceDataset(Dataset):
    """
    Dataset used for inference on all tiles corresponding to a single map.  Used by MobileNetRCNN.
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def __init__(
        self,
        tiles: List[ImageTile],
    ) -> None:
        self.tiles = tiles
        super().__init__()

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, Dict]:
        image = np.array(self.tiles[idx].image)  # type: ignore
        image = torch.tensor(image).float().permute(2, 0, 1).to(self.device)
        metadata = {
            "x_offset": self.tiles[idx].x_offset,  # type: ignore
            "y_offset": self.tiles[idx].y_offset,  # type: ignore
            "width": self.tiles[idx].width,  # type: ignore
            "height": self.tiles[idx].height,  # type: ignore
            "map_path": self.tiles[idx].map_path,
        }  # type: ignore

        return image, metadata
