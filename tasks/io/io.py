from pathlib import Path
import os
from typing import Iterator, List, Tuple
from PIL.Image import Image as PILImage
from PIL import Image


class ImageFileInputIterator(Iterator[Tuple[str, PILImage]]):
    """Generates an iterable list of PIL images from a directory of images"""

    def __init__(self, image_path: Path) -> None:
        """Initializes the iterator"""
        self._image_files: List[Path] = []
        self._index = 0

        # recursivley traverse the input directory and find all image files
        for root, _, files in os.walk(image_path):
            for file in files:
                self._image_files.append(Path(os.path.join(root, file)))

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[str, PILImage]:
        """Loads the next image in the list of images"""
        if self._index < len(self._image_files):
            image_path = self._image_files[self._index]
            image = Image.open(image_path)
            doc_id = image_path.stem
            self._index += 1
            return (doc_id, image)
        else:
            raise StopIteration
