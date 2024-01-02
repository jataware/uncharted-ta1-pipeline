import logging
import re
import json
from typing import List, Optional, Dict, Any
from enum import Enum
from openai import OpenAI
import tiktoken
from tasks.common.task import TaskInput, TaskResult
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.common.pipeline import Task
from typing import List, Dict, Any

logger = logging.getLogger("metadata_extractor")


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
            "quadrangle": "<quadrangle>",
        },
        indent=4,
    )

    def __init__(self, id: str, model=LLM.GPT_3_5_TURBO):
        super().__init__(id)
        self._openai_client = (
            OpenAI()
        )  # will read key from "OPENAI_API_KEY" env variable
        self._model = model
        logger.info(f"Using model: {self._model.value}")

    def run(self, input: TaskInput) -> TaskResult:
        """Processes a directory of OCR files and writes the metadata to a json file"""
        # extract metadata from ocr output
        text_data = input.data[TEXT_EXTRACTION_OUTPUT_KEY]
        doc_text = DocTextExtraction.model_validate(text_data)
        task_result = TaskResult(self._task_id)

        metadata = self._process_doc_text_extraction(doc_text)
        if metadata:
            # normalize scale
            metadata.scale = self._normalize_scale(metadata.scale)

            # normalize quadrangle
            metadata.quadrangle = self._normalize_quadrangle(metadata.quadrangle)
            task_result.add_output(
                METADATA_EXTRACTION_OUTPUT_KEY, metadata.model_dump()
            )

        return task_result

    def _process_doc_text_extraction(
        self, doc_text_extraction: DocTextExtraction, model: str = "gpt-3.5-turbo"
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
                        "content": "You are using text extracted from US geological maps by an OCR process to identify map metadata",
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
            + " Find the map title, scale, projection, geoditic datum, vertical datum, coordinate systems, authors, year, base map, quadrangle\n"
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
            + "A singel author is allowed.\n"
            + "Authors, title and year are normally grouped together.\n"
            + "The year should be the most recent value and should be a single 4 digit number.\n"
            + "The term grid ticks should not be included in coordinate system output.\n"
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

    def _normalize_quadrangle(self, quadrangle_str: str) -> str:
        """Normalizes the quadrangle string by removing the word quadrangle"""
        return re.sub(self.QUADRANGLE_PATTERN, "", quadrangle_str).strip()

    @staticmethod
    def _create_empty_extraction(doc_id: str) -> MetadataExtraction:
        """Creates an empty metadata extraction object"""
        return MetadataExtraction(
            map_id=doc_id,
            title="",
            authors=[],
            year="",
            scale="",
            quadrangle="",
            datum="",
            vertical_datum="",
            projection="",
            coordinate_systems=[],
            base_map="",
        )
