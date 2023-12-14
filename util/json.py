import json

from typing import Any, Dict


def read_json_file(path: str) -> Dict[Any, Any]:
    # read it as json
    raw_file = open(path)
    return json.load(raw_file)
