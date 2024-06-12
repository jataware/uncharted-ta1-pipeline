import copy
import logging
import re
import json
from enum import Enum
from openai import OpenAI
import cv2
import numpy as np
from PIL.Image import Image as PILImage
import tiktoken
from tasks.common.image_io import pil_to_cv_image
from tasks.common.task import TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MapChromaType,
    MapShape,
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.segmentation.entities import SEGMENTATION_OUTPUT_KEY, MapSegmentation
from tasks.text_extraction.entities import (
    DocTextExtraction,
    TextExtraction,
    TEXT_EXTRACTION_OUTPUT_KEY,
)
from tasks.common.pipeline import Task
from typing import Callable, List, Dict, Any, Optional, Tuple

logger = logging.getLogger("metadata_extractor")

PLACE_EXTENSION_MAP = {"washington": "washington (state)"}


class LLM(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"

    def __str__(self):
        return self.value


class MetadataExtractor(Task):
    # matcher for alphanumeric strings
    ALPHANUMERIC_PATTERN = re.compile(r".*[a-zA-Z].*\d.*|.*\d.*[a-zA-Z].*|.*[a-zA-Z].*")

    # patterns for scale normalization
    SCALE_PATTERN = re.compile(r"[,\. a-zA-z]+")
    SCALE_PREPEND = re.compile(r"\d+:")

    # quadrangle normalization
    QUADRANGLE_PATTERN = re.compile(re.escape("quadrangle"), re.IGNORECASE)

    # max number of tokens allowed by openai api, leaving enough for output
    TOKEN_LIMIT = 3500

    # OCR text filtering control
    MAX_TEXT_FILTER_LENGTH = 600
    MIN_TEXT_FILTER_LENGTH = 100
    TEXT_FILTER_DECREMENT = 100

    # json structure for prompt
    EXAMPLE_JSON = json.dumps(
        {
            "title": "<title>",
            "projection": "<projection>",
            "scale": "<scale>",
            "datum": "<datum>",
            "vertical_datum": "<vertical datum>",
            "coordinate_systems": [
                "<coordinate system>",
                "<coordinate_system>",
                "<coordinate_system>",
            ],
            "utm_zone": "<utm zone>",
            "authors": ["<author name>", "<author name>", "<author name>"],
            "year": "<publication year>",
            "publisher": "<publisher>",
            "base_map": "<base map>",
            "counties": ["<county>", "<county>", "<county>"],
            "states": ["<state>", "<state>", "<state>"],
            "country": "<country>",
            "publisher": "<publisher>",
            "language": "<language>",
            "language_country": "<language country>",
        },
        indent=4,
    )
    EXAMPLE_JSON_POINTS = json.dumps(
        [
            {"name": "Ducky Hill", "index": 12},
            {"name": "Bear Mtn", "index": 34},
            {"name": "Bedford Hill", "index": 20},
            {"name": "Spear Peak", "index": 6},
        ],
        indent=4,
    )

    EXAMPLE_JSON_CITIES = json.dumps(
        [
            {"name": "Denver", "index": 12},
            {"name": "Los Angeles", "index": 34},
            {"name": "Seattle", "index": 20},
            {"name": "Calgary", "index": 6},
        ],
        indent=4,
    )

    EXAMPLE_JSON_UTM = json.dumps({"utm_zone": "<utm zone>"})

    EXAMPLE_JSON_QUADRANGLES = json.dumps(
        {"quadrangles": ["<quadrangle>", "<quadrangle>"]}
    )

    # threshold for determining map shape - anything above is considered rectangular
    RECTANGULARITY_THRESHOLD = 0.9

    def __init__(
        self,
        id: str,
        model=LLM.GPT_4_TURBO,
        text_key=TEXT_EXTRACTION_OUTPUT_KEY,
        should_run: Optional[Callable] = None,
    ):
        super().__init__(id)
        self._openai_client = (
            OpenAI()
        )  # will read key from "OPENAI_API_KEY" env variable
        self._model = model
        self._text_key = text_key
        self._should_run = should_run
        logger.info(f"Using model: {self._model.value}")

    def run(self, input: TaskInput) -> TaskResult:
        """Processes a directory of OCR files and writes the metadata to a json file"""
        if self._should_run and not self._should_run(input):
            return self._create_result(input)

        task_result = TaskResult(self._task_id)

        # check if metadata already exists
        metadata_raw = input.get_data(METADATA_EXTRACTION_OUTPUT_KEY)
        if metadata_raw:
            metadata = MetadataExtraction.model_validate(metadata_raw)
            # add the place extraction
            # TODO: THIS IS A TEMPORARY HACK UNTIL REFACTORED TO WORK WITH LANGCHAIN
            doc_text: DocTextExtraction = input.parse_data(
                TEXT_EXTRACTION_OUTPUT_KEY, DocTextExtraction.model_validate
            )

            text_indices = self._extract_text_with_index(doc_text)

            # convert text to prompt string and compute token count
            prompt_str_places = self._to_point_prompt_str(
                self._text_extractions_to_str(text_indices)
            )
            metadata.places = self._process_map_area_extractions(
                doc_text, prompt_str_places
            )
            prompt_str_areas = self._to_place_prompt_str(
                self._text_extractions_to_str(text_indices)
            )
            metadata.population_centres = self._process_map_area_extractions(
                doc_text, prompt_str_areas, True
            )
            task_result.add_output(
                METADATA_EXTRACTION_OUTPUT_KEY, metadata.model_dump()
            )
            return task_result

        # extract metadata from ocr output
        doc_text: DocTextExtraction = input.parse_data(
            self._text_key, DocTextExtraction.model_validate
        )
        if not doc_text:
            return task_result

        # post-processing and follow on prompts
        metadata = self._process_doc_text_extraction(doc_text)
        if metadata:
            # map state names as needed
            for i, p in enumerate(metadata.states):
                if p.lower() in PLACE_EXTENSION_MAP:
                    metadata.states[i] = PLACE_EXTENSION_MAP[p.lower()]

            # normalize scale
            metadata.scale = self._normalize_scale(metadata.scale)

            # normalize quadrangle
            metadata.quadrangles = self._normalize_quadrangle(metadata.quadrangles)

            # extract quadrangles from the title and base map info
            metadata.quadrangles = self._extract_quadrangles(
                metadata.title, metadata.base_map
            )

            # extract UTM zone if not present in metadata after initial extraction
            if metadata.utm_zone == "NULL":
                metadata.utm_zone = self._extract_utm_zone(metadata)

            # compute map shape from the segmentation output
            segments = input.data[SEGMENTATION_OUTPUT_KEY]
            metadata.map_shape = self._compute_shape(segments)

            # compute map chroma from the image
            metadata.map_chroma = self._compute_chroma(input.image)

            task_result.add_output(
                METADATA_EXTRACTION_OUTPUT_KEY, metadata.model_dump()
            )

        return task_result

    def _process_doc_text_extraction(
        self, doc_text_extraction: DocTextExtraction
    ) -> Optional[MetadataExtraction]:
        """Extracts metadata from a single OCR file"""
        try:
            logger.info(f"Processing '{doc_text_extraction.doc_id}'")

            max_text_length = self.MAX_TEXT_FILTER_LENGTH
            num_tokens = 0
            prompt_str = ""
            text = []
            while max_text_length > self.MIN_TEXT_FILTER_LENGTH:
                # extract text from OCR output using rule-based filtering
                text = self._extract_text(doc_text_extraction, max_text_length)

                # convert text to prompt string and compute token count
                prompt_str = self._to_prompt_str("\n".join(text))
                num_tokens = self._count_tokens(prompt_str, "cl100k_base")

                # if the token count is greater than the limit, reduce the max text length
                # and try again
                if num_tokens <= self.TOKEN_LIMIT:
                    break
                max_text_length = max_text_length - self.TEXT_FILTER_DECREMENT
                logger.info(
                    f"Token count after filtering exceeds limit - reducing max text length to {max_text_length}"
                )

            logger.info(f"Processing {num_tokens} tokens.")

            logger.debug("Prompt string:\n")
            logger.debug(prompt_str)

            if num_tokens < self.TOKEN_LIMIT:
                messages: List[Any] = [
                    {
                        "role": "system",
                        "content": "You are using text extracted from maps by an OCR process to identify map metadata",
                    },
                    {"role": "user", "content": prompt_str},
                ]
                # GPT-4-turbo allows for an explicit response format setting for JSON output
                if self._model == LLM.GPT_4_TURBO:
                    completion = self._openai_client.chat.completions.create(
                        messages=messages,
                        model=self._model.value,
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                else:
                    completion = self._openai_client.chat.completions.create(
                        messages=messages,
                        model=self._model.value,
                        temperature=0.1,
                    )

                message_content = completion.choices[0].message.content
                if message_content is not None:
                    try:
                        content_dict: Dict[str, Any] = json.loads(message_content)
                        country = content_dict["country"]
                        if country is None or len(country) == 0 or country == "NULL":
                            content_dict["country"] = self._get_country("\n".join(text))
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Skipping extraction '{doc_text_extraction.doc_id}' - error parsing json response from openai api likely due to token limit",
                            exc_info=True,
                        )
                        return self._create_empty_extraction(doc_text_extraction.doc_id)
                    content_dict["map_id"] = doc_text_extraction.doc_id
                else:
                    content_dict = {"map_id": doc_text_extraction.doc_id}
                # ensure all the dict values are populated so we can validate the model - those below
                # are extracted by the first GPT pass
                # TODO - is it possible to handle this in a better way?
                content_dict["places"] = []
                content_dict["population_centres"] = []
                content_dict["map_shape"] = MapShape.UNKNOWN
                content_dict["map_chroma"] = MapChromaType.UNKNOWN
                content_dict["vertical_datum"] = "NULL"
                content_dict["quadrangles"] = []
                extraction = MetadataExtraction(**content_dict)
                return extraction

            logger.warn(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - input token count {num_tokens} is greater than limit {self.TOKEN_LIMIT}"
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

        except Exception as e:
            logger.error(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - unexpected error during processing",
                exc_info=True,
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

    def _get_country(self, text_str: str) -> Optional[str]:
        country_prompt = self._to_country_prompt_str(text_str)
        messages: List[Any] = [
            {
                "role": "system",
                "content": "You are using text extracted from maps by an OCR process to identify map metadata",
            },
            {"role": "user", "content": country_prompt},
        ]
        # GPT-4-turbo allows for an explicit response format setting for JSON output
        if self._model == LLM.GPT_4_TURBO:
            completion = self._openai_client.chat.completions.create(
                messages=messages,
                model=self._model.value,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        else:
            completion = self._openai_client.chat.completions.create(
                messages=messages,
                model=self._model.value,
                temperature=0.1,
            )
        message_content = completion.choices[0].message.content
        return message_content

    def _process_map_area_extractions(
        self,
        doc_text_extraction: DocTextExtraction,
        prompt_str: str,
        replace_text: bool = False,
    ) -> List[TextExtraction]:
        logger.info(
            f"extracting point places from the map area of '{doc_text_extraction.doc_id}'"
        )
        places = []
        try:
            messages: List[Any] = [
                {
                    "role": "system",
                    "content": "You are using text extracted from US geological maps by an OCR process to identify map metadata",
                },
                {"role": "user", "content": prompt_str},
            ]
            if self._model == LLM.GPT_4_TURBO:
                completion = self._openai_client.chat.completions.create(
                    messages=messages,
                    model=self._model.value,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            else:
                completion = self._openai_client.chat.completions.create(
                    messages=messages,
                    model=self._model.value,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

            message_content = completion.choices[0].message.content
            if message_content is not None:
                try:
                    places_raw: List[Dict[str, Any]] = json.loads(message_content)[
                        "points"
                    ]
                    places = self._map_text_coordinates(
                        places_raw, doc_text_extraction, replace_text
                    )
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Skipping extraction '{doc_text_extraction.doc_id}' - error parsing json response from api likely due to token limit",
                        exc_info=True,
                    )
                    places = []
        except Exception as e:
            logger.error(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - unexpected error during processing",
                exc_info=True,
            )
            places = []
        return places

    def _extract_utm_zone(self, metadata: MetadataExtraction) -> str:
        """Extracts the UTM zone from the metadata if it is not already present"""
        prompt_str = self._to_utm_prompt_str(
            metadata.counties,
            metadata.quadrangles,
            metadata.states,
            metadata.places,
            metadata.population_centres,
        )
        utm_zone_resp = self._process_basic_prompt(prompt_str)
        if utm_zone_resp == "NULL":
            return utm_zone_resp
        utm_json = json.loads(utm_zone_resp)
        return utm_json["utm_zone"]

    def _extract_quadrangles(self, title: str, base_map: str) -> List[str]:
        """Extracts quadrangles from the title and base map info"""
        prompt_str = self._to_quadrangle_prompt_str(title, base_map)
        quadrangle_resp = self._process_basic_prompt(prompt_str)
        if quadrangle_resp == "NULL":
            return []
        quadrangle_json = json.loads(quadrangle_resp)
        return quadrangle_json["quadrangles"]

    def _process_basic_prompt(self, prompt_str: str) -> str:
        message_content: str | None = ""
        try:
            messages: List[Any] = [
                {
                    "role": "system",
                    "content": "You are using text extracted from US geological maps by an OCR process to identify map metadata",
                },
                {"role": "user", "content": prompt_str},
            ]
            if self._model == LLM.GPT_4_TURBO:
                completion = self._openai_client.chat.completions.create(
                    messages=messages,
                    model=self._model.value,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )
            else:
                completion = self._openai_client.chat.completions.create(
                    messages=messages,
                    model=self._model.value,
                    response_format={"type": "json_object"},
                    temperature=0.1,
                )

            message_content = completion.choices[0].message.content
            if message_content is not None:
                try:
                    result = message_content.strip()
                except json.JSONDecodeError as e:
                    logger.error(
                        f"Skipping extraction - error parsing json response from api likely due to token limit",
                        exc_info=True,
                    )
        except Exception as e:
            logger.error(
                f"Skipping extraction - unexpected error during processing",
                exc_info=True,
            )
        return "" if message_content is None else message_content

    def _extract_text_with_index(
        self, doc_text_extraction: DocTextExtraction
    ) -> List[Tuple[str, int]]:
        # map all text with index
        return [(d.text, i) for i, d in enumerate(doc_text_extraction.extractions)]

    def _text_extractions_to_str(self, extractions: List[Tuple[str, int]]) -> str:
        # want to end up with a list of (text, coordinate) having each entry be a new line
        items = [f"({r[0]}, {i})" for i, r in enumerate(extractions)]

        return "\n".join(items)

    def _map_text_coordinates(
        self,
        places: List[Dict[str, Any]],
        extractions: DocTextExtraction,
        replace_text: bool,
    ) -> List[TextExtraction]:
        # want to use the index to filter the extractions
        # TODO: MAY WANT TO CHECK THE TEXT LINES UP JUST IN CASE THE LLM HAD A BIT OF FUN
        filtered = []
        for p in places:
            e = copy.deepcopy(extractions.extractions[p["index"]])
            if replace_text:
                e.text = p["name"]
            if "state" in p:
                e.text = f"{e.text}, {p['state']}"
            filtered.append(e)
        return filtered  # type: ignore

    def _count_tokens(self, input_str: str, encoding_name: str) -> int:
        """Counts the number of tokens in a input string using a given encoding"""
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(input_str))
        return num_tokens

    def _to_prompt_str(self, text_str: str) -> str:
        """Converts a string of text to a prompt string for GPT-3.5-turbo"""
        return (
            "The following blocks of text were extracted from a map using an OCR process:\n"
            + text_str
            + "\n\n"
            + " Find the following:\n"
            + " - map title. \n"
            + " - scale\n"
            + " - projection\n"
            + " - datum\n"
            + " - vertical datum\n"  # explicitly look for this so we can ignore it - if we totally remove we start to see vertical datums show up as datums
            + " - coordinate systems\n"
            + " - UTM zone\n"
            + " - authors\n"
            + " - year\n"
            + " - base map description or base map (quad, year pairs)\n"
            + " - counties\n"
            + " - states/provinces\n"
            + " - country\n"
            + " - map publisher\n"
            + " - language\n"
            + " - language country\n"
            + " Examples of geoditic datums: North American Datum of 1983, NAD83, WGS 84.\n"
            + " Examples of vertical datums: 'Mean Sea Level', 'Mean Low Water', and 'national vertical geoditic datum of 1929'\n"  # explicitly look for this so we can ignore it
            + " Examples of UTM zones: 12, 15.\n"
            + " Examples of projections: Polyconic, Lambert, Transverse Mercator\n"
            + " Examples of coordinate systems: Utah coordinate system central zone, UTM Zone 15, Universal Transverse Mercator zone 12, New Mexico coordinate system, north zone, Carter Coordinate System"
            + ' Examples of base maps: "U.S. Geological Survey 1954", "U.S. Geological Survey 1:62,500, Vidal (1949) Rice and Turtle Mountains (1954) Savahia Peak (1975)"\n'
            + ' Examples of states/provinces: "Arizona", "New York", "South Dakota", "Ontario"\n'
            + " Return the data as a JSON structure.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON
            + "\n"
            + 'If any string value is not present the field should be set to "NULL".\n'
            + "If the map title includes state names, county names or quadrangle, they should be included in the full title.\n"
            + "All author names should be in the format: <last name, first iniital, middle initial>.  Example of author name: Bailey, D. K.\n"
            + "A single author is allowed.  Note that authors, title and year are normally grouped together on the map so use that help disambiguate.\n"
            + "References, citations and geology attribution should be ignored when extracting authors.\n"
            + "The year should be the most recent value and should be a single 4 digit number.\n"
            + "Geoditic datums should be in their short form, for example, North American Datum of 1927 should be NAD27.\n"
            + "Geoditic datums are exclusive of vertical datums; geoditic datums never contain terms associated with height or verticality.\n"
            + "The term grid ticks should not be included in coordinate system output.\n"
            + "The base map description can be a descriptive string, but also often contains (quadrangle, year) pairs.\n"
            + "States includes principal subvidisions of any country and their full name should be extracted.\n"
            + "UTM zones should not include an N or S after the number when extracted.\n"
            + "Language should capture the best guess of the language used in the blocks of text.\n"
            + "Language country should be the name of the country derived from Language.\n"
        )

    def _to_point_prompt_str(self, text_str: str) -> str:
        return (
            "The following blocks of text were extracted from a map using an OCR process, specified as a list with (text, index):\n"
            + text_str
            + "\n\n"
            + " Return the places that are points. The following types of places are points: \n"
            + " - mountains\n"
            + " - peaks\n"
            + " - trailheads\n"
            + " - hills\n"
            + " - summits\n\n"
            + " Ignore places that are not points. The following types of places are not points: \n"
            + " - pond\n"
            + " - brook\n"
            + " - lake\n"
            + " - river\n\n"
            + " In the response, include the index and the name as part of a tuple in a json formatted list with each item being {name: 'name', index: 'index'}.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON_POINTS
            + "\n"
        )

    def _to_place_prompt_str(self, text_str: str) -> str:
        return (
            "The following blocks of text were extracted from a map using an OCR process, specified as a list with (text, index):\n"
            + text_str
            + "\n\n"
            + " Return the places that are recognizable metropolitan areas, cities, towns, or villages.\n"
            + " Ignore places that are roads, streets, avenues, or other similar features.\n"
            + " Using the above list of places, determine which state or province is most likely being explored.\n"
            + " In the response, include the index, the name and the state or province as part of a tuple in a json formatted list with each item being {name: 'name', index: 'index', state: 'state'}.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON_CITIES
            + "\n\n"
            + ' In the returned json, name the result "points".'
        )

    def _to_country_prompt_str(self, text_str: str) -> str:
        return (
            "The following blocks of text were extracted from a map using an OCR process:\n"
            + text_str
            + "\n\n"
            + " Return the most likely country of the map. It could be derived from place names, states, or the language of the map.\n"
            + " Return the result as a simple string."
        )

    def _to_utm_prompt_str(
        self,
        counties: List[str],
        quadrangles: List[str],
        state: List[str],
        places: List[TextExtraction],
        population_centers: List[TextExtraction],
    ) -> str:
        return (
            "The following information was extracted froma map using an OCR process:\n"
            + f"quadrangles: {','.join(quadrangles)}\n"
            + f"counties: {','.join(counties)}\n"
            + f"places: {','.join([p.text for p in places])}\n"
            + f"population centers: {','.join([p.text for p in population_centers])}\n"
            + f"states: {','.join(state)}\n"
            + " Infer the UTM zone and return it in a JSON structure. If it cannot be inferred, return 'NULL'.\n"
            + " The inferred UTM zone should not include an N or S after the number.\n"
            + " Here is an example of the structure to return: \n"
            + self.EXAMPLE_JSON_UTM
        )

    def _to_quadrangle_prompt_str(self, title: str, base_map_info: str) -> str:
        return (
            "The following information was extracted from a map using an OCR process:\n"
            + f"title: {title}\n"
            + f"base map: {base_map_info}\n"
            + " Identify the quadrangles from the fields and store them in a JSON structure.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON_QUADRANGLES
            + "\n"
        )

    def _extract_text(
        self, doc_text_extraction: DocTextExtraction, max_text_length=800
    ) -> List[str]:
        """Extracts text from OCR output - filters to alphanumeric strings between 4 and 400 characters long
        that contain at least one space"""
        text_dims = []
        for text_entry in doc_text_extraction.extractions:
            text = text_entry.text
            if (
                self.ALPHANUMERIC_PATTERN.match(text)
                and len(text) >= 4
                and len(text) <= max_text_length
                and len(text.split(" ")) > 1
            ):
                text_dims.append(text)
        return text_dims

    def _normalize_scale(self, scale_str: str) -> str:
        """Normalizes the scale string to the format 1:xxxxx"""
        if scale_str != "NULL":
            normalized_scale = re.sub(self.SCALE_PATTERN, "", scale_str)
            if not re.match(self.SCALE_PREPEND, normalized_scale):
                normalized_scale = "1:" + normalized_scale
            return normalized_scale
        return scale_str

    def _normalize_quadrangle(self, quadrangles_str: List[str]) -> List[str]:
        """Normalizes the quadrangle string by removing the word quadrangle"""
        return [
            re.sub(self.QUADRANGLE_PATTERN, "", quad_str).strip()
            for quad_str in quadrangles_str
        ]

    def _compute_shape(self, segments) -> MapShape:
        """
        Computes the shape of the map from the segmentation output using a rectangularity metric

        Args:
            segments: The segmentation output

        Returns:
            MapShape: The shape of the map
        """
        if segments:
            map_segmentation = MapSegmentation.model_validate(segments)
            for segment in map_segmentation.segments:
                if segment.class_label == "map":
                    box_area = (segment.bbox[2] - segment.bbox[0]) * (
                        segment.bbox[3] - segment.bbox[1]
                    )
                    rectangularity = segment.area / box_area
                    if rectangularity > self.RECTANGULARITY_THRESHOLD:
                        map_shape = MapShape.RECTANGULAR
                    else:
                        map_shape = MapShape.IRREGULAR
                    break
        return map_shape

    def _compute_chroma(
        self, input_image: PILImage, max_dim=500, mono_thresh=20, low_thresh=60
    ) -> MapChromaType:
        """
        Computes the chroma of the map image using the LAB color space
        and the centroid of the a and b channels

        Args:
            input_image (PILImage): The map image
            max_dim (int): The maximum dimension for resizing the image

        Returns:
            MapChromaType: The chroma type of the map
        """
        if max_dim > 0:
            # uniformly resize the image so that major axis is max_dim
            image = pil_to_cv_image(input_image)
            h, w, _ = image.shape
            if h > w:
                image = cv2.resize(image, (max_dim, int(h / w * max_dim)))
            else:
                image = cv2.resize(image, (int(w / h * max_dim), max_dim))

        # exract the a and b channels and find the centroid
        cs_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        cs_image = cs_image[:, :, 1:].flatten().reshape(-1, 2)

        # compute the error between the mean and each pixel
        mean_vec = np.sum(cs_image, axis=0) / len(cs_image)
        dist = np.linalg.norm(cs_image - mean_vec, axis=1)

        # square the distance and take the mean
        error = np.mean(dist**2)

        # classify the chroma based on the error
        if error < mono_thresh:
            return MapChromaType.MONO_CHROMA
        elif error < low_thresh:
            return MapChromaType.LOW_CHROMA
        else:
            return MapChromaType.HIGH_CHROMA

    @staticmethod
    def _create_empty_extraction(doc_id: str) -> MetadataExtraction:
        """Creates an empty metadata extraction object"""
        return MetadataExtraction(
            map_id=doc_id,
            title="",
            authors=[],
            year="",
            scale="",
            quadrangles=[],
            datum="",
            projection="",
            coordinate_systems=[],
            utm_zone="",
            base_map="",
            counties=[],
            states=[],
            population_centres=[],
            country="",
            places=[],
            publisher="",
            map_shape=MapShape.UNKNOWN,
            map_chroma=MapChromaType.UNKNOWN,
            language="",
            language_country="",
        )
