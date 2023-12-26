import logging
import re
import pprint
import json
from tasks.common.task import TaskInput, TaskResult
from openai import OpenAI
import tiktoken
from typing import List, Optional, Dict, Any
from tasks.metadata_extraction.entities import (
    MetadataExtraction,
    METADATA_EXTRACTION_OUTPUT_KEY,
)
from tasks.text_extraction.entities import DocTextExtraction, TEXT_EXTRACTION_OUTPUT_KEY
from tasks.common.pipeline import Task


logger = logging.getLogger("metadata_extractor")


class MetadataExtractor(Task):
    # matcher for alphanumeric strings
    ALPHANUMERIC_PATTERN = re.compile(r".*[a-zA-Z].*\d.*|.*\d.*[a-zA-Z].*|.*[a-zA-Z].*")

    # patterns for scale normalization
    SCALE_PATTERN = re.compile(r"[,\. a-zA-z]+")
    SCALE_PREPEND = re.compile(r"\d+:")

    # quadrangle normalization
    QUADRANGLE_PATTERN = re.compile(re.escape("quadrangle"), re.IGNORECASE)

    # max number of tokens allowed by openai api
    TOKEN_LIMIT = 4096

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

    def __init__(
        self,
        id: str,
        verbose=False,
    ):
        super().__init__(id)
        self._verbose = verbose
        self._openai_client = (
            OpenAI()
        )  # will read key from "OPENAI_API_KEY" env variable

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
        self, doc_text_extraction: DocTextExtraction
    ) -> Optional[MetadataExtraction]:
        """Extracts metadata from a single OCR file"""
        try:
            text = [text_entry.text for text_entry in doc_text_extraction.extractions]
            prompt_str = self._to_prompt_str("\n".join(text))
            num_tokens = self._count_tokens(prompt_str, "cl100k_base")

            logger.info(f"Processing '{doc_text_extraction.doc_id}'")
            logger.info(f"Found {num_tokens} tokens.")

            logger.debug("Prompt string:\n")
            logger.debug(prompt_str)

            if num_tokens < self.TOKEN_LIMIT:
                completion = self._openai_client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are using text extracted from US geological maps by an OCR process to identify map metadata",
                        },
                        {"role": "user", "content": prompt_str},
                    ],
                    model="gpt-3.5-turbo",
                    temperature=0.1,
                )

                message_content = completion.choices[0].message.content
                if message_content is not None:
                    content_dict: Dict[str, Any] = json.loads(message_content)
                    content_dict["map_id"] = doc_text_extraction.doc_id
                else:
                    content_dict = {"map_id": doc_text_extraction.doc_id}
                extraction = MetadataExtraction(**content_dict)
                return extraction

            logger.warn(
                "skipping extraction '{doc_text_extraction.doc_id}' exceeded to token limit"
            )
            return None

        except Exception as e:
            # print exception stack trace
            logger.exception(
                f"Error: An exception occurred while processing '{doc_text_extraction.doc_id}'",
                exc_info=True,
            )
            return None

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
            + " Examples of coordinate systems: Utah coordinate system central zone, UTM Zone 15, Universal Transverse Mercator zone 12, New Mexico coordinate system, north zone\n"
            + ' Examples of base maps: "U.S. Geological Survey 1954", "U.S. Geological Survey 1:62,500, Vidal (1949) Rice and Turtle Mountains (1954) Savahia Peak (1975)"\n'
            + " Return the data as a JSON structure.\n"
            + " Here is an example of the structure to use: \n"
            + self.EXAMPLE_JSON
            + "\n"
            + 'If any string value is not present the field should be set to "NULL"\n'
            + "All author names should be in the format: <last name, first iniital, middle initial>.  Example of author name: Bailey, D. K.\n"
            + "References and citations should be ignored when extracting authors.\n"
            + "Authors, title and year are normally grouped together.\n"
            + "The year should be the most recent value and should be a single 4 digit number.\n"
            + "The term grid ticks should not be included in coordinate system output.\n"
        )

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
