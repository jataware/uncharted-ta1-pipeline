from ast import parse
import copy
from dataclasses import dataclass
import logging
import re
from enum import Enum
from langchain_openai import ChatOpenAI
import cv2
import numpy as np
from PIL.Image import Image as PILImage
from langchain.schema import SystemMessage, PromptValue
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic.v1 import BaseModel, Field
import tiktoken
from tasks.common.image_io import pil_to_cv_image
from tasks.common.task import TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MapColorType,
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
import hashlib


logger = logging.getLogger("metadata_extractor")

PLACE_EXTENSION_MAP = {"washington": "washington (state)"}


class LLM(str, Enum):
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO = "gpt-4-turbo"
    GPT_4 = "gpt-4"
    GPT_4_O = "gpt-4o"

    def __str__(self):
        return self.value


class MetdataLLM(BaseModel):
    title: str = Field(
        description="The title of the map. If this includes state names, "
        + "county names or quadrangles, still include them in full title. "
        + " Example: 'Geologic map of the Grand Canyon Quadrangle, Arizona'",
        default="NULL",
    )
    authors: List[str] = Field(
        description="The authors of the map. "
        + "Should be a list of strings in the format <last name, first iniital, middle initial>"
        + "Example of author name: 'Bailey, D. K.'"
        + "References, citations and geology attribution should be ignored when extracting authors."
        + "A single author is allowed. Authors, title and year are normally grouped together.",
        default=[],
    )
    year: str = Field(
        description="The year the map was published"
        + "Should be a single 4 digit number and the most recent year if multiple are present",
        default="NULL",
    )
    scale: str = Field(
        description="The scale of the map.  Example: '1:24000'", default="NULL"
    )
    datum: str = Field(
        description="The geoditic datum of the map. If this is not present, it can often be inferred from the map's country and year."
        + "Examples of geodetic datums: 'North American Datum of 1927', 'NAD83', 'WGS 84'",
        default="NULL",
    )
    vertical_datum: str = Field(
        description="The vertical datum of the map."
        + "Examples: 'mean sea level', 'vertical datum of 1901', 'national vertical geodetic datum of 1929'",
        default="NULL",
    )
    projection: str = Field(
        description="The map projection."
        + "Examples: 'Polyconic', 'Lambert', 'Transverse Mercator'",
        default="NULL",
    )
    coordinate_systems: List[str] = Field(
        description="The coordinate systems present on the map."
        + "Examples: 'Utah coordinate system central zone', 'UTM Zone 15', "
        + "'Universal Transverse Mercator zone 12', "
        + "'New Mexico coordinate system, north zone', 'Carter Coordinate System'."
        + "The term `grid ticks` should not be included in coordinate system output.",
        default=[],
    )
    base_map: str = Field(
        description="The base map information description.  The base map description can be a "
        + "descriptive string, but also often contains (quadrangle, year) pairs."
        + "Examples: 'U.S. Geological Survey 1954', "
        + "'U.S. Geological Survey 1:62,500', "
        + "'Vidal (1949) Rice and Turtle Mountains (1954) Savahia Peak (1975)'",
        default="NULL",
    )
    utm_zone: int = Field(
        description="The UTM zone of the map.  If the UTM zone cannot be inferred, return 0.",
        default=0,
    )

    counties: List[str] = Field(
        description="Counties covered by the map.  These are often listed in the title; "
        + "if they are not, they should be extracted from the map.",
        default=[],
    )
    states: List[str] = Field(
        description="Principal subdivisions (eg. states, provinces) covered by the map, expressed using ISO 3166-2 codes."
        + " Examples: 'US-AZ', 'US-NY', 'CA-ON'",
        default=[],
    )
    country: str = Field(
        description="Country covered by the map expressed using ISO 3166-1 codes."
        + "Examples: 'US', 'CA', 'GB'",
        default="NULL",
    )
    publisher: str = Field(description="The publisher of the map.", default="NULL")
    language: str = Field(
        description="The best guess of the language used in the blocks of text.",
        default="NULL",
    )
    language_country: str = Field(
        description="The name of the country derived from the extracted Language.",
        default="NULL",
    )


class Location(BaseModel):
    name: str = Field(
        description="The name of the location extracted from the map area. "
        + "The name should be the name of the point and the index of the point in the extracted text.",
        default="NULL",
    )
    index: int = Field(
        description="The index of the point in the extracted text.", default=-1
    )


class PointLocationsLLM(BaseModel):
    # points: List[Tuple[str, int]] = Field(
    places: List[Location] = Field(
        description="The list of point places extracted from the map area. "
        + "The 'name' key should contain the name of the point and the 'index' key should contain the index of "
        + "the point in the extracted text.  Point places are land based features that are not population centers."
        + "Examples of places that are points: mountains, peaks, trailheads, hills, summits.\n"
        + "Examples of places that are not points: pond, brook, lake, river.\n",
        default=[],
    )
    population_centers: List[Location] = Field(
        description="The list of recognizeable population centers extracted from the map area. "
        + "The 'name' key should contain the name of the population center and the 'index' key should contain the "
        + "index of the population center in the extracted text."
        + "Examples of population centers: cities, towns, villages, hamlets.\n"
        + "Examples of places that are not population centers: roads, streets, avenues, or other similar features.\n",
        default=[],
    )


@dataclass
class PointLocations:
    places: List[TextExtraction]
    population_centers: List[TextExtraction]


class UTMZoneLLM(BaseModel):
    utm_zone: int = Field(description="The UTM zone of the map", default=0)


class QuadranglesLLM(BaseModel):
    quadrangles: List[str] = Field(
        description="The list of quadrangles extracted from the map area.", default=[]
    )


class StateCountryLLM(BaseModel):
    states: List[str] = Field(
        description="Principal subdivisions (eg. states, provinces) covered by the map, expressed using ISO 3166-2 codes."
        + " Examples: 'US-AZ', 'US-NY', 'CA-ON",
        default=[],
    )
    country: str = Field(
        description="Country covered by the map expressed using ISO 3166-1 codes."
        + "Examples: 'US', 'CA', 'GB'",
        default="NULL",
    )


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

    TEXT_EXTRACT_TEMPLATE = (
        "The following blocks of text were extracted from a map using an OCR process:\n"
        + "{text_str}"
        + "\n\n"
        + "Extract metadata defined in the output structure from the text.\n"
        + "{format}"
        + "\n"
        + 'If any string value is not present the field should be set to "NULL"\n'
    )

    POINT_LOCATIONS_TEMPLATE = (
        "The following blocks of text were extracted from a map using an OCR process, specified "
        + "as a list of tuples with (text, index):\n"
        + "{text_str}"
        + "\n\n"
        + "Extract the places that are map point features and population centers from the text.\n"
        + "{format}"
    )

    UTM_ZONE_TEMPLATE = (
        "The following information was extracted froma map using an OCR process:\n"
        + "coordinate systems: {coordinate_systems}"
        + "quadrangles: {quadrangles}\n"
        + "counties: {counties}\n"
        + "places: {places}\n"
        + "population centers: {population_centers}\n"
        + "states: {states}\n"
        + "country {country}\n"
        + "Infer the UTM zone from the above information. If it cannot be inferred, return 0.\n"
        + "{format}"
    )

    QUADRANGLES_TEMPLATE = (
        "The following information was extracted from a map using an OCR process:\n"
        + "title: {title}\n"
        + "base map: {base_map}\n"
        + "Identify the quadrangles from the fields above.\n"
        + "{format}"
    )

    STATE_COUNTRY_TEMPLATE = (
        "The following information was extracted from a map using an OCR process:\n"
        + "population centers: {population_centers}\n"
        + "places: {places}\n"
        + "counties: {counties}\n"
        + "Infer the states and country from the fields above.\n"
        + "{format}"
    )

    # threshold for determining map shape - anything above is considered rectangular
    RECTANGULARITY_THRESHOLD = 0.9

    def __init__(
        self,
        id: str,
        model=LLM.GPT_4_O,
        text_key=TEXT_EXTRACTION_OUTPUT_KEY,
        should_run: Optional[Callable] = None,
        cache_dir: str = "",
        include_place_bounds: bool = True,
    ):
        super().__init__(id, cache_location=cache_dir)

        self._chat_model = ChatOpenAI(
            model=model, temperature=0.1
        )  # reads OPEN_AI_API_KEY from environment
        self._model = model
        self._text_key = text_key
        self._include_place_bounds = include_place_bounds
        self._should_run = should_run

        logger.info(f"Using model: {self._model.value}")

    def run(self, input: TaskInput) -> TaskResult:
        """
        Runs the metadata extraction task.

        Args:
            input (TaskInput): The input data for the task.

        Returns:
            TaskResult: The result of the task.
        """

        logger.info(f"Running metadata extraction task for '{input.raster_id}'")

        if self._should_run and not self._should_run(input):
            logging.info("Skipping metadata extraction task")
            return self._create_result(input)

        task_result = TaskResult(self._task_id)
        doc_text: DocTextExtraction = input.parse_data(
            self._text_key,
            DocTextExtraction.model_validate,
            self._create_empty_extraction,
        )
        if not doc_text:
            logger.info(
                "OCR output not available - returning empty metadata extraction result"
            )
            return task_result

        doc_id = self._generate_doc_key(input, doc_text)

        # use the cached result if available
        result = self.fetch_cached_result(doc_id)
        if result:
            metadata = MetadataExtraction.model_validate(result)
            logger.info(f"Using cached metadata extraction result for key {doc_id}")
            task_result.add_output(METADATA_EXTRACTION_OUTPUT_KEY, result)
            return task_result

        logger.info(f"No cached metadata extraction result found for {doc_id}")

        # post-processing and follow on prompts
        metadata: Optional[MetadataExtraction] = input.parse_data(
            METADATA_EXTRACTION_OUTPUT_KEY, MetadataExtraction.model_validate
        )
        if metadata is None:
            metadata = self._process_doc_text_extraction(doc_text)

        if metadata:
            logger.info("Post-processing metadata extraction result")
            # map state names as needed
            for i, p in enumerate(metadata.states):
                if p.lower() in PLACE_EXTENSION_MAP:
                    metadata.states[i] = PLACE_EXTENSION_MAP[p.lower()]

            # normalize scale
            metadata.scale = self._normalize_scale(metadata.scale)

            # # extract places
            point_locations = self._extract_point_locations(doc_text)
            if not self._include_place_bounds:
                metadata.places = [p.text for p in point_locations.places]
                metadata.population_centres = [
                    p.text for p in point_locations.population_centers
                ]
            else:
                metadata.places = point_locations.places
                metadata.population_centres = point_locations.population_centers

            # extract state and country if not present in metadata after initial extraction
            if not metadata.states or metadata.country == "NULL":
                metadata.states, metadata.country = self._extract_state_country(
                    metadata
                )

            # extract quadrangles
            metadata.quadrangles = self._extract_quadrangles(metadata)

            # extract UTM zone if not present in metadata after initial extraction
            if int(metadata.utm_zone) == 0:
                metadata.utm_zone = str(self._extract_utm_zone(metadata))

            # compute map shape from the segmentation output
            segments = input.data.get(SEGMENTATION_OUTPUT_KEY, None)
            metadata.map_shape = self._compute_shape(segments)

            # compute map chroma from the image
            metadata.map_color = self._compute_color_level(input.image)

            # update the cache
            self.write_result_to_cache(metadata, doc_id)

            task_result.add_output(
                METADATA_EXTRACTION_OUTPUT_KEY, metadata.model_dump()
            )
        else:
            logger.warn("No metadata extraction result found")

        return task_result

    def _generate_doc_key(
        self, task_input: TaskInput, doc_text: DocTextExtraction
    ) -> str:
        """
        Generates a unique document key based on the given task input and configuration.

        Args:
            task_input (TaskInput): The input for the task.

        Returns:
            str: The generated document key.
        """
        attributes = "_".join(
            [
                "metadata",
                task_input.raster_id,
                self._model,
                str(self._include_place_bounds),
                doc_text.model_dump_json(),
            ]
        )
        doc_key = hashlib.sha256(attributes.encode()).hexdigest()
        return doc_key

    def _process_doc_text_extraction(
        self, doc_text_extraction: DocTextExtraction
    ) -> Optional[MetadataExtraction]:
        """
        Processes the text extraction from the OCR output and extracts metadata from it using an LLM .

        Args:
            doc_text_extraction (DocTextExtraction): The text extraction from the OCR output

        Returns:
            Optional[MetadataExtraction]: The extracted metadata
        """

        try:
            logger.info(
                f"Processing doc text extractions from '{doc_text_extraction.doc_id}'"
            )

            max_text_length = self.MAX_TEXT_FILTER_LENGTH
            num_tokens = 0

            input_prompt: Optional[PromptValue] = None
            text = []

            # setup the output structure
            parser = PydanticOutputParser(pydantic_object=MetdataLLM)

            # setup the prompt template
            prompt_template = self._generate_prompt_template(
                parser, self.TEXT_EXTRACT_TEMPLATE
            )

            while max_text_length > self.MIN_TEXT_FILTER_LENGTH:
                # extract text from OCR output using rule-based filtering
                text = self._extract_text(doc_text_extraction, max_text_length)
                input_prompt = prompt_template.format_prompt(text_str="\n".join(text))
                if input_prompt is None:
                    logger.warn(
                        f"Skipping extraction '{doc_text_extraction.doc_id}' - prompt generation failed"
                    )
                    return self._create_empty_extraction(doc_text_extraction.doc_id)

                # if the token count is greater than the limit, reduce the max text length
                # and try again
                num_tokens = self._count_tokens(input_prompt.to_string(), "cl100k_base")
                if num_tokens <= self.TOKEN_LIMIT:
                    break
                max_text_length = max_text_length - self.TEXT_FILTER_DECREMENT
                logger.debug(
                    f"Token count after filtering exceeds limit - reducing max text length to {max_text_length}"
                )
            logger.info(f"Processing {num_tokens} tokens.")

            # generate the response
            if input_prompt is not None:
                logger.debug("Prompt string:\n")
                logger.debug(input_prompt.to_string())
                chain = prompt_template | self._chat_model | parser
                response = chain.invoke({"text_str": "\n".join(text)})
                # add placeholders for fields we don't extract
                response_dict = response.dict()
                response_dict["quadrangles"] = []
                response_dict["population_centres"] = []
                response_dict["places"] = []
                response_dict["map_shape"] = "unknown"
                response_dict["map_color"] = "unknown"
                return MetadataExtraction(
                    map_id=doc_text_extraction.doc_id, **response_dict
                )

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

    def _extract_point_locations(
        self,
        doc_text: DocTextExtraction,
    ) -> PointLocations:
        """
        Uses an LLM to extract point locations from input texts.

        Args:
            doc_text (DocTextExtraction): The document text to extract locations from.

        Returns:
            List[TextExtraction]: A list of extracted locations as TextExtraction objects.
        """
        logger.info(f"Secondary extraction of point locations")

        text_indices = self._extract_text_with_index(doc_text)

        parser = PydanticOutputParser(pydantic_object=PointLocationsLLM)
        prompt_template = self._generate_prompt_template(
            parser, self.POINT_LOCATIONS_TEMPLATE
        )
        chain = prompt_template | self._chat_model | parser
        response: PointLocationsLLM = chain.invoke({"text_str": text_indices})

        places = self._map_text_coordinates(response.places, doc_text, True)
        population_centers = self._map_text_coordinates(
            response.population_centers, doc_text, True
        )
        return PointLocations(places=places, population_centers=population_centers)

    def _extract_state_country(
        self, metadata: MetadataExtraction
    ) -> Tuple[List[str], str]:
        """
        Extracts the state and country from the given metadata.

        Args:
            metadata (MetadataExtraction): The metadata containing information about states and countries.

        Returns:
            Tuple[str, str]: A tuple containing the extracted state and country.
        """
        logger.info("Secondary extraction of state/country")

        parser = PydanticOutputParser(pydantic_object=StateCountryLLM)
        prompt_template = self._generate_prompt_template(
            parser, self.STATE_COUNTRY_TEMPLATE
        )
        chain = prompt_template | self._chat_model | parser
        response: StateCountryLLM = chain.invoke(
            {
                "places": " ".join(
                    [
                        s.text if isinstance(s, TextExtraction) else s
                        for s in metadata.places
                    ]
                ),
                "population_centers": " ".join(
                    s.text if isinstance(s, TextExtraction) else s
                    for s in metadata.population_centres
                ),
                "counties": metadata.counties,
            }
        )
        return (response.states, response.country)

    def _extract_utm_zone(self, metadata: MetadataExtraction) -> int:
        """
        Infers the UTM zone from the given metadata using an LLM.

        Args:
            metadata (MetadataExtraction): The metadata containing information about counties, quadrangles, states,
                places, and population centers.

        Returns:
            int: The UTM zone extracted from the metadata. 0 indicates that the UTM zone could not be inferred.
        """
        logger.info(f"Secondary extraction of UTM zone")

        args = {
            "coordinate_systems": ",".join(metadata.coordinate_systems),
            "counties": ",".join(metadata.counties),
            "quadrangles": ",".join(metadata.quadrangles),
            "states": ",".join(metadata.states),
            "country": metadata.country,
            "places": ",".join(
                [
                    s.text if isinstance(s, TextExtraction) else s
                    for s in metadata.places
                ]
            ),
            "population_centers": ",".join(
                s.text if isinstance(s, TextExtraction) else s
                for s in metadata.population_centres
            ),
        }
        parser = PydanticOutputParser(pydantic_object=UTMZoneLLM)
        prompt_template = self._generate_prompt_template(parser, self.UTM_ZONE_TEMPLATE)
        chain = prompt_template | self._chat_model | parser
        response = chain.invoke(args)

        return response.utm_zone

    def _extract_quadrangles(self, metadata: MetadataExtraction) -> List[str]:
        """
        Infers quadrangles from the given metadata using an LLM.

        Args:
            metadata (MetadataExtraction): The metadata object containing the necessary information.

        Returns:
            List[str]: A list of extracted quadrangles.
        """
        logger.info(f"Secondary extraction of quadrangles")

        args = {"title": metadata.title, "base_map": metadata.base_map}

        parser = PydanticOutputParser(pydantic_object=QuadranglesLLM)
        prompt_template = self._generate_prompt_template(
            parser, self.QUADRANGLES_TEMPLATE
        )
        chain = prompt_template | self._chat_model | parser
        response = chain.invoke(args)

        return self._normalize_quadrangles(response.quadrangles)

    def _extract_text_with_index(
        self, doc_text_extraction: DocTextExtraction
    ) -> List[Tuple[str, int]]:
        """
        Extracts the text from the given `doc_text_extraction` object along with their respective indices.

        Args:
            doc_text_extraction (DocTextExtraction): The `DocTextExtraction` object containing the text extractions.

        Returns:
            List[Tuple[str, int]]: A list of tuples where each tuple contains the extracted text and its index.
        """
        # map all text with index
        return [(d.text, i) for i, d in enumerate(doc_text_extraction.extractions)]

    def _text_extractions_to_str(self, extractions: List[Tuple[str, int]]) -> str:
        """
        Converts a list of text extractions to a string representation.

        Args:
            extractions (List[Tuple[str, int]]): A list of tuples containing the extracted text and its coordinate.

        Returns:
            str: A string representation of the text extractions, with each entry on a new line.
        """
        items = [f"({r[0]}, {i})" for i, r in enumerate(extractions)]
        return "\n".join(items)

    def _map_text_coordinates(
        self,
        places: List[Location],
        extractions: DocTextExtraction,
        replace_text: bool,
    ) -> List[TextExtraction]:
        """
        Maps the text coordinates of the given places to the corresponding extractions.

        Args:
            places (List[Location]): The list of locations to map.
            extractions (DocTextExtraction): The document text extractions.
            replace_text (bool): Flag indicating whether to replace the text in the extractions with the name of the place.

        Returns:
            List[TextExtraction]: The filtered list of text extractions.

        """
        # want to use the index to filter the extractions
        # TODO: MAY WANT TO CHECK THE TEXT LINES UP JUST IN CASE THE LLM HAD A BIT OF FUN
        filtered = []
        for p in places:
            e = copy.deepcopy(extractions.extractions[p.index])
            if replace_text:
                e.text = p.name
            filtered.append(e)
        return filtered  # type: ignore

    def _count_tokens(self, input_str: str, encoding_name: str) -> int:
        """
        Counts the number of tokens in the input string using the specified encoding.

        Args:
            input_str (str): The input string to count tokens from.
            encoding_name (str): The name of the encoding to use.

        Returns:
            int: The number of tokens in the input string.

        """
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(input_str))
        return num_tokens

    def _generate_prompt_template(self, parser, template: str) -> ChatPromptTemplate:
        """
        Generates a chat prompt template from an input string.

        Args:
            parser (Parser): The parser object used for extracting metadata.
            template (str): The template string for the human message.

        Returns:
            ChatPromptTemplate: The generated chat prompt template.
        """
        system_message = "You are using text extracted from geological maps by an OCR process to identify map metadata"
        human_message_template = HumanMessagePromptTemplate.from_template(template)
        prompt = ChatPromptTemplate(
            messages=[
                SystemMessage(content=system_message),
                human_message_template,
            ],
            input_variables=["text_str"],
            partial_variables={"format": parser.get_format_instructions()},
        )
        return prompt

    def _extract_text(
        self, doc_text_extraction: DocTextExtraction, max_text_length=800
    ) -> List[str]:
        """
        Extracts text from OCR output - filters to alphanumeric strings between 4 and 400 characters long
        that contain at least one space

        Args:
            doc_text_extraction (DocTextExtraction): The text extraction from the OCR output
            max_text_length (int): The maximum length of the text to extract

        Returns:
            List[str]: The extracted text
        """
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
        """
        Normalizes the scale string to the format 1:xxxxx

        Args:
            scale_str (str): The scale string to normalize

        Returns:
            str: The normalized scale string
        """
        if scale_str != "NULL":
            normalized_scale = re.sub(self.SCALE_PATTERN, "", scale_str)
            if not re.match(self.SCALE_PREPEND, normalized_scale):
                normalized_scale = "1:" + normalized_scale
            return normalized_scale
        return scale_str

    def _normalize_quadrangles(self, quadrangles_str: List[str]) -> List[str]:
        """
        Normalizes the quadrangle string by removing the word quadrangle

        Args:
            quadrangles_str (List[str]): The quadrangle strings to normalize

        Returns:
            List[str]: The normalized quadrangle strings
        """
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
        map_shape = MapShape.UNKNOWN
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

    def _compute_color_level(
        self, input_image: PILImage, max_dim=500, mono_thresh=20, low_thresh=60
    ) -> MapColorType:
        """
        Computes the colour level of the map image using the LAB color space
        and the centroid of the a and b channels

        Args:
            input_image (PILImage): The map image
            max_dim (int): The maximum dimension for resizing the image

        Returns:
            MapColorType: The chroma type of the map
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
            return MapColorType.MONO
        elif error < low_thresh:
            return MapColorType.LOW
        else:
            return MapColorType.HIGH

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
            map_color=MapColorType.UNKNOWN,
            language="",
            language_country="",
        )
