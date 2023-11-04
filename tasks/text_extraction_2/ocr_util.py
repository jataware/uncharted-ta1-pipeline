import io
import os
import pickle
import copy
import numpy as np
from typing import List, Dict, Any, Tuple
from google.cloud import vision
from pathlib import Path
from google.cloud.vision import Image as VisionImage, AnnotateImageResponse
from PIL import Image, ImageDraw
from PIL.Image import Image as PILImage
from PIL.ImageDraw import ImageDraw as PILImageDraw
from tqdm import tqdm

# NOTE: env var GOOGLE_APPLICATION_CREDENTIALS must be set to the path of the service account key

PIXEL_LIM = 8000

Image.MAX_IMAGE_PIXELS = 400000000  # to allow PIL to load large images


def load_vision_image(path: str) -> VisionImage:
    """Loads an image into memory as a google vision api image object"""
    with io.open(path, "rb") as image_file:
        content = image_file.read()
    return VisionImage(content=content)


def load_pil_image(path: str) -> PILImage:
    """Loads an image into memory as a PIL image object"""
    return Image.open(path)


def pil_to_vision_image(pil_image: PILImage) -> VisionImage:
    """Converts a PIL image object to a google vision api image object"""
    pil_image_bytes = io.BytesIO()
    pil_image.save(pil_image_bytes, format="JPEG")
    return VisionImage(content=pil_image_bytes.getvalue())


def condition_pil_image(pil_image: PILImage, max_dim=PIXEL_LIM) -> PILImage:
    """Resizes the image so that no axis is larger than max_dim pixels and
    ensures the pixel format is compatible with the google vision api"""
    image_orig_size = pil_image.size
    image_resize_ratio = 1.0

    if pil_image.mode == "F":
        # floating point pixel format, need to convert to uint8
        np_image = np.asarray(pil_image)
        pxl_mult = 255 / max(1.0, np.max(np_image))
        print(
            "WARNING! Raw image is in 32-bit floating point format. Scaling by {} and converting to uint8".format(
                pxl_mult
            )
        )
        pil_image = Image.fromarray(np.uint8(np_image * pxl_mult))
    if pil_image.mode in ("RGBA", "P"):
        pil_image = pil_image.convert("RGB")

    if max(image_orig_size) > max_dim:
        image_resize_ratio = max_dim / max(image_orig_size)

        reduced_size = int(image_orig_size[0] * image_resize_ratio), int(
            image_orig_size[1] * image_resize_ratio
        )
        pil_image = pil_image.resize(reduced_size, Image.Resampling.LANCZOS)

    return pil_image


def load_ocr_output(input_path: str) -> List[Dict[str, Any]]:
    """Loads the text annotation"""
    with open(input_path, "rb") as f:
        return pickle.load(f)


def detect_text(vision_api_image: VisionImage) -> List[Dict[str, Any]]:
    """Runs google vision OCR on the image and returns the text annotations as a dictionary"""
    client = vision.ImageAnnotatorClient()
    response = client.text_detection(image=vision_api_image)  # type: ignore
    texts = []
    if response.text_annotations:  # first entry will be the entire text block
        texts = [
            {"text": text.description, "bounding_poly": text.bounding_poly}
            for text in response.text_annotations
        ]
    else:
        print("WARNING! No OCR text found!")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )

    return texts


def text_to_blocks(texts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Groups OCR text into blocks based on the bounding polygons"""
    if len(texts) < 2:
        print(
            "WARNING! less than 2 blocks of OCR text found! Skipping ocr_text_to_blocks"
        )
        return []

    full_text = texts[0]["text"]

    group_offset = 1
    num_blocks = 0
    group_counter = 0

    results: List[Dict[str, Any]] = []

    text_blocks = full_text.split("\n")
    text_block = text_blocks[0]

    for text_block_next in text_blocks[1:]:
        text_block = text_block.strip()
        text_block0 = text_block
        bounding_poly = None  # vision.BoundingPoly()
        for text in texts[group_offset:]:
            prose = text["text"].strip()
            group_counter += 1
            text_block_sub = text_block.replace(
                prose, "", 1
            ).strip()  # TODO could make this replace from the start of string...
            if len(text_block_sub) == 0:
                bounding_poly = add_bounding_polygons(
                    bounding_poly, text["bounding_poly"]
                )
                break
            elif (
                len(prose) > 0
                and len(text_block_sub) == len(text_block)
                and text_block_next.startswith(prose)
            ):  # and len(text_block_sub) < 3
                # partial OCR block parsed! ... finish with this block and go to next one
                group_counter -= 1
                break
            bounding_poly = add_bounding_polygons(bounding_poly, text["bounding_poly"])
            text_block = text_block_sub

        num_blocks += 1

        if bounding_poly is not None:
            results.append({"text": text_block0, "bounding_poly": bounding_poly})
        group_offset += group_counter
        group_counter = 0
        text_block = text_block_next

    # and save last block too
    # results.append({'text' : text_block.strip(), 'bounding_poly' : bounding_poly})
    if len(results) < len(text_blocks) - 1:
        print(
            "WARNING! Possible error grouping OCR results"
        )  # TODO - throw exception here?

    return results


def add_bounding_polygons(poly1: Any, poly2: Any) -> Any:
    """Adds two bounding polygons together to create a new bounding polygon that contains both"""
    if poly1 is None:
        return poly2
    if poly2 is None:
        return poly1

    min_x = poly1.vertices[0].x
    min_y = poly1.vertices[0].y
    max_x = poly1.vertices[0].x
    max_y = poly1.vertices[0].y

    for vertex in poly1.vertices:
        min_x = min(min_x, vertex.x)
        min_y = min(min_y, vertex.y)
        max_x = max(max_x, vertex.x)
        max_y = max(max_y, vertex.y)

    for vertex in poly2.vertices:
        min_x = min(min_x, vertex.x)
        min_y = min(min_y, vertex.y)
        max_x = max(max_x, vertex.x)
        max_y = max(max_y, vertex.y)

    poly_combined = copy.deepcopy(poly1)
    poly_combined.vertices[0].x = min_x
    poly_combined.vertices[1].x = max_x
    poly_combined.vertices[2].x = max_x
    poly_combined.vertices[3].x = min_x

    poly_combined.vertices[0].y = min_y
    poly_combined.vertices[1].y = min_y
    poly_combined.vertices[2].y = max_y
    poly_combined.vertices[3].y = max_y

    return poly_combined


def detect_document_text(vision_api_image: VisionImage) -> List[Dict[str, Any]]:
    """Runs google vision OCR on the image and returns the text annotations as a dictionary"""
    texts = []
    client = vision.ImageAnnotatorClient()
    response = client.document_text_detection(image=vision_api_image)  # type: ignore
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_words = []
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        word_text = "".join([symbol.text for symbol in word.symbols])
                        block_words.append(word_text)

                texts.append(
                    {
                        "text": " ".join(block_words),
                        "bounding_poly": block.bounding_box,
                    }
                )
    else:
        print("WARNING! No OCR text found!")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )
    return texts


def write_texts(texts: List[Dict[str, Any]], path: str):
    """Writes the text annotations to a json file"""
    with open(path, "wb") as f:
        pickle.dump(texts, f)


def write_reponse(response: AnnotateImageResponse, path: str):
    """Serializes the text protobuf annotation response to a file"""
    with open(path, "wb") as f:
        serialized_proto = AnnotateImageResponse.serialize(response)
        f.write(serialized_proto)


def display_ocr_results(texts: List[Dict[str, Any]], pil_image: PILImage, color="red"):
    """Draws the bounding polygons of each text box on the image and displays it"""
    draw_img = ImageDraw.Draw(pil_image)
    for text in texts:
        draw_img.polygon(
            [(vertex.x, vertex.y) for vertex in text["bounding_poly"].vertices],
            outline=color,
        )
    pil_image.show()


def process_images(
    input_path: Path, output_path: Path, to_blocks=False, document_ocr=False
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Processes all images in a directory and writes the text annotations out as pkl files"""

    results: List[Tuple[str, List[Dict[str, Any]]]] = []
    for _, _, filenames in os.walk(input_path):
        for filename in tqdm(filenames):
            file = os.path.join(input_path, filename)

            # determine the output file name
            output_file = os.path.join(
                output_path, os.path.splitext(filename)[0] + ".pkl"
            )

            doc_id = Path(file).with_suffix("").stem

            # skip if the output file already exists
            if os.path.isfile(output_file):
                with open(output_file, "rb") as f:
                    results.append((doc_id, pickle.load(f)))
                    continue

            try:
                if os.path.isfile(file):
                    image = load_pil_image(file)
                    conditioned_image = condition_pil_image(image)
                    vision_image = pil_to_vision_image(conditioned_image)
                    if document_ocr:
                        texts = detect_document_text(vision_image)
                    else:
                        texts = detect_text(vision_image)
                        if to_blocks:
                            texts = text_to_blocks(texts)

                    write_texts(texts, output_file)
                    results.append((doc_id, texts))
            except Exception as e:
                print(f"Error processing file: {file} - {e}")
                continue
    return results


def process_image(
    input_path: Path, output_path: Path, to_blocks=False, document_ocr=False
) -> Tuple[str, List[Dict[str, Any]]]:
    """Processes a single image and writes the text annotations to a json file"""
    # determine the output file name
    output_file = os.path.join(output_path, input_path.with_suffix("").stem + ".pkl")

    doc_id = input_path.with_suffix("").stem

    # skip if the output file already exists
    if os.path.isfile(output_file):
        with open(output_file, "rb") as f:
            return (doc_id, pickle.load(f))

    image = load_pil_image(str(input_path))
    conditioned_image = condition_pil_image(image)
    vision_image = pil_to_vision_image(conditioned_image)
    if document_ocr:
        texts = detect_document_text(vision_image)
    else:
        texts = detect_text(vision_image)
        if to_blocks:
            texts = text_to_blocks(texts)
    write_texts(texts, str(output_file))
    return (doc_id, texts)
