import copy
import logging
import re
import json
from enum import Enum
from openai import OpenAI
import tiktoken
from tasks.common.task import TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
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
    GPT_4_TURBO = "gpt-4-1106-preview"
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
            "authors": ["<author name>", "<author name>", "<author name>"],
            "year": "<publication year>",
            "publisher": "<publisher>",
            "base_map": "<base map>",
            "quadrangles": ["<quadrangle>", "<quadrangle>", "<quadrangle>"],
            "counties": ["<county>", "<county>", "<county>"],
            "states": ["<state>", "<state>", "<state>"],
            "country": "<country>",
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

    def __init__(
        self,
        id: str,
        model=LLM.GPT_3_5_TURBO,
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

        # extract metadata from ocr output
        text_data = input.data[self._text_key]
        doc_text = DocTextExtraction.model_validate(text_data)
        task_result = TaskResult(self._task_id)

        # check if metadata already exists
        metadata_raw = input.get_data(METADATA_EXTRACTION_OUTPUT_KEY)
        if metadata_raw:
            metadata = MetadataExtraction.model_validate(metadata_raw)
            text_data = input.data[TEXT_EXTRACTION_OUTPUT_KEY]
            doc_text = DocTextExtraction.model_validate(text_data)
            # add the place extraction
            # TODO: THIS IS A TEMPORARY HACK UNTIL REFACTORED TO WORK WITH LANGCHAIN
            doc_text = DocTextExtraction.model_validate(text_data)

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
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Skipping extraction '{doc_text_extraction.doc_id}' - error parsing json response from openai api likely due to token limit",
                            exc_info=True,
                        )
                        return self._create_empty_extraction(doc_text_extraction.doc_id)
                    content_dict["map_id"] = doc_text_extraction.doc_id
                else:
                    content_dict = {"map_id": doc_text_extraction.doc_id}
                content_dict["places"] = []
                content_dict["population_centres"] = []
                extraction = MetadataExtraction(**content_dict)
                return extraction

            logger.warn(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - input token count {num_tokens} is greater than limit {self.TOKEN_LIMIT}"
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

        except Exception as e:
            # print exception stack trace
            logger.error(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - unexpected error during processing",
                exc_info=True,
            )
            return self._create_empty_extraction(doc_text_extraction.doc_id)

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
            # print exception stack trace
            logger.error(
                f"Skipping extraction '{doc_text_extraction.doc_id}' - unexpected error during processing",
                exc_info=True,
            )
            places = []
        return places

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
            + " - map title\n"
            + " - scale\n"
            + " - projection\n"
            + " - geoditic datum\n"
            + " - vertical datum\n"
            + " - coordinate systems\n"
            + " - authors\n"
            + " - year\n"
            + " - base map info\n"
            + " - quadrangles\n"
            + " - counties\n"
            + " - states\n"
            + " - country\n"
            + " Examples of vertical datums: mean sea level, vertical datum of 1901\n"
            + " Examples of datums: North American Datum of 1927, NAD83, WGS 84\n"
            + " Examples of projections: Polyconic, Lambert, Transverse Mercator\n"
            + " Examples of coordinate systems: Utah coordinate system central zone, UTM Zone 15, Universal Transverse Mercator zone 12, New Mexico coordinate system, north zone, Carter Coordinate System"
            + ' Examples of base maps: "U.S. Geological Survey 1954", "U.S. Geological Survey 1:62,500, Vidal (1949) Rice and Turtle Mountains (1954) Savahia Peak (1975)"\n'
            + " Return the data as a JSON structure.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON
            + "\n"
            + 'If any string value is not present the field should be set to "NULL"\n'
            + "All author names should be in the format: <last name, first iniital, middle initial>.  Example of author name: Bailey, D. K.\n"
            + "References, citations and geology attribution should be ignored when extracting authors.\n"
            + "A single author is allowed.\n"
            + "Authors, title and year are normally grouped together.\n"
            + "The year should be the most recent value and should be a single 4 digit number.\n"
            + "The term grid ticks should not be included in coordinate system output.\n"
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
            vertical_datum="",
            projection="",
            coordinate_systems=[],
            base_map="",
            counties=[],
            states=[],
            population_centres=[],
            country="",
            places=[],
        )
