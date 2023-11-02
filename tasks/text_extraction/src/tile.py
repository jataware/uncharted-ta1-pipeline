from PIL import Image

class Tile():
    '''
    Tile representing a partial image
    '''

    # TODO - could be moved to a 'common' module?

    image: Image.Image = None
    coordinates: tuple[int] = (0,0)

    def __init__(self, image: Image.Image, coordinates: tuple[int]):
        self.image = image
        self.coordinates = coordinates