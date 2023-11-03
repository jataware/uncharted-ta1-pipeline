import re
import openai
import os
import pprint
import json
import tiktoken
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Optional
from entities import MetadataExtraction
from tasks.text_extraction_2.entities import DocTextExtraction

# env var for openai api key
openai.api_key = os.environ["OPENAI_API_KEY"]


class MetadataExtractor:
    # matcher for alphanumeric strings
    ALPHANUMERIC_PATTERN = re.compile(r".*[a-zA-Z].*\d.*|.*\d.*[a-zA-Z].*|.*[a-zA-Z].*")

    # patterns for scale normalization
    SCALE_PATTERN = re.compile(r"[,\. a-zA-z]+")
    SCALE_PREPEND = re.compile(r"\d+:")

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
            "lat_lon": "<latitude and longitude point on map>",
        },
        indent=4,
    )

    def __init__(
        self,
        doc_text_extractions: List[DocTextExtraction],
        output_path: Path,
        verbose=False,
    ):
        self._doc_text_extractions = doc_text_extractions
        self._output_path = output_path
        self._verbose = verbose

    def process(self) -> List[MetadataExtraction]:
        """Extracts metadata from OCR output"""
        # if the output dir doesn't exist, create it
        if not self._output_path.exists():
            os.makedirs(self._output_path)
        return self._process_maps(self._doc_text_extractions, self._verbose)

    def _process_maps(
        self,
        doc_text_extractions: List[DocTextExtraction],
        verbose=False,
    ) -> List[MetadataExtraction]:
        """Processes a directory of OCR files and writes the metadata to a json file"""
        metadata_list: List[MetadataExtraction] = []
        for doc_text in tqdm(doc_text_extractions):
            # extract metadata from ocr output
            map_metadata = self._process_doc_text_extractions(doc_text, verbose=verbose)
            if map_metadata:
                # normalize scale
                if map_metadata:
                    map_metadata.scale = self._normalize_scale(map_metadata.scale)
                metadata_list.append(map_metadata)

        return metadata_list

    def _process_doc_text_extractions(
        self, doc_text_extraction: DocTextExtraction, verbose=False
    ) -> Optional[MetadataExtraction]:
        """Extracts metadata from a single OCR file"""
        try:
            text = self._extract_text(doc_text_extraction, verbose=verbose)
            prompt_str = self._to_prompt_str("\n".join(text))

            num_tokens = self._count_tokens(prompt_str, "cl100k_base")

            if verbose:
                print("Prompt string:\n")
                print(prompt_str)
                print(f"Found {num_tokens} tokens.")

            if num_tokens < self.TOKEN_LIMIT:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    # model="gpt-4",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are using text extracted from US geological maps by an OCR process to identify map metadata",
                        },
                        {"role": "user", "content": prompt_str},
                    ],
                    temperature=0.1,
                )

                content_str = response["choices"][0]["message"]["content"]  # type: ignore
                content_dict = json.loads(content_str)
                content_dict["map_id"] = doc_text_extraction.doc_id
                extraction = MetadataExtraction(**content_dict)
                return extraction

            return None

        except Exception as e:
            # print exception stack trace
            print(
                f"Error: An exception occurred while processing '{doc_text_extraction.doc_id}': {str(e)}"
            )
            return None

    def _extract_text(
        self, doc_text_extraction: DocTextExtraction, verbose=False
    ) -> List[str]:
        """Extracts text from OCR output"""
        text_dims = []
        for text_entry in doc_text_extraction.extractions:
            text = text_entry.text
            if (
                self.ALPHANUMERIC_PATTERN.match(text)
                and len(text) >= 4
                and len(text) <= 400
                and len(text.split(" ")) > 1
            ):
                text_dims.append(text)

        if verbose:
            print("Extracted text:\n")
            pprint.pprint(text_dims)
            print()
        return text_dims

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
            + 'The term "grid ticks" is not part of the coordinate system.\n'
        )

    def _normalize_scale(self, scale_str: str) -> str:
        """Normalizes the scale string to the format 1:xxxxx"""
        if scale_str != "NULL":
            normalized_scale = re.sub(self.SCALE_PATTERN, "", scale_str)
            if not re.match(self.SCALE_PREPEND, normalized_scale):
                normalized_scale = "1:" + normalized_scale
            return normalized_scale
        return scale_str


class MetadataFileWriter:
    def __init__(self, metadata: List[MetadataExtraction], output_path: Path):
        self._metadata = metadata
        self._output_path = output_path

    def process(self) -> None:
        """Writes metadata to a json file"""
        # if the output dir doesn't exist, create it
        if not self._output_path.exists():
            output_dir = os.path.dirname(self._output_path)
            os.makedirs(output_dir)

        # write metadata to json file
        for metadata in self._metadata:
            json_model = metadata.model_dump_json()
            with open(
                os.path.join(self._output_path, metadata.map_id + ".json"), "w"
            ) as outfile:
                outfile.write(json_model)
