import datetime
import json
from typing import Any, Dict


class JSONLog:
    def __init__(self, file: str):
        self._file = file

    def log(self, log_type: str, data: Dict[str, Any]):
        # append the data as json, treating the file as a json lines file
        log_data = {
            "timestamp": f"{datetime.datetime.now()}",
            "log_type": log_type,
            "data": data,
        }
        with open(self._file, "a") as log_file:
            log_file.write(f"{json.dumps(log_data)}\n")
