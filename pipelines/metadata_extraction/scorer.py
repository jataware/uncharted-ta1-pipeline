from typing import List, Dict, Union
from pathlib import Path
import json
from nltk.translate.meteor_score import single_meteor_score
import os
from tasks.entities.entities import MetadataExtraction

MetadataScore = Dict[str, float]
MetadataScores = Dict[str, MetadataScore]


class Scorer:
    def __init__(self, truth: Path, predictions: Path, verbose: bool = False):
        self._truth = truth
        self._predictions = predictions
        self._verbose = verbose
        self._results: MetadataScores = {}

    def score(self) -> None:
        file_type = self._truth.suffix
        if file_type == ".jsonl":
            truth = self._load_json(self._truth)
        else:
            print("error - invalid file type")
            exit()

        for map_key, map_value in truth.items():
            prediction_filename = f"{map_key}.json"
            prediction_path = Path(os.path.join(self._predictions, prediction_filename))
            if prediction_path.exists():
                with open(prediction_path, "r") as predictions:
                    predictions = MetadataExtraction(**json.load(predictions))
                    self._results[map_key] = self._process_json(
                        map_value, predictions, self._verbose
                    )

    def print_results(self) -> None:
        """print results to stdout"""
        for map_key in self._results.keys():
            for field_key in self._results[map_key].keys():
                print(f"{map_key},{field_key},{self._results[map_key][field_key]}")
            print("===============================")

    def print_scores(self) -> None:
        """print the collected results"""
        print("Results:\n====================")
        total_score = 0.0

        field_scores: Dict[str, List[float]] = {}
        for key, value in self._results.items():
            for field_key, field_score in value.items():
                if field_key not in field_scores.keys():
                    field_scores[field_key] = []
                field_scores[field_key].append(field_score)

        for field_key, score_list in field_scores.items():
            # loop over the value list and count the number of times value is 1.0
            mean_score = sum(score_list) / len(score_list)

            print(field_key)
            print(f"Samples: {len(score_list)}")
            print(f"Mean score: {mean_score}")
            print()

    def _load_json(self, path: Path) -> Dict[str, MetadataExtraction]:
        """load ground truth from a jsonl file"""
        json_data: Dict[str, MetadataExtraction] = {}

        with open(path, "r") as json_file:
            for line in json_file:
                data = json.loads(line)
                metadata = MetadataExtraction(**data)
                json_data[metadata.map_id] = metadata
            return json_data

    def _process_json(
        self,
        truth: MetadataExtraction,
        predictions: MetadataExtraction,
        verbose: bool = False,
    ) -> MetadataScore:
        """compare ground truth to predictions"""
        results: MetadataScore = {}

        # loop over each field in the map
        for field_key in predictions:
            if field_key[0] in truth.model_fields_set:
                # score the prediction
                field_truth = getattr(truth, field_key[0])
                # skip predictions that are null / empty when scoring
                if (
                    isinstance(field_truth, str)
                    and (field_truth == "NULL" or len(field_truth) == 0)
                ) or (
                    isinstance(field_truth, list)
                    and len(field_truth) > 0
                    and field_truth[0] == "NULL"
                ):
                    continue

                field_prediction = getattr(predictions, field_key[0])

                score = 0.0
                if isinstance(field_prediction, list) and isinstance(field_truth, list):
                    score = self._score_list(field_prediction, field_truth)
                elif isinstance(field_prediction, str) and isinstance(field_truth, str):
                    if field_key == "title":
                        score = self._score_meteor(field_prediction, field_truth)
                    else:
                        score = self._score_string(field_prediction, field_truth)
                else:
                    score = self._score_string(str(field_prediction), str(field_truth))

                if verbose:
                    print(f" Field: {field_key}")
                    print(f" Predicted: {field_prediction}")
                    print(f" Truth: {field_truth}")
                    print(f" Score: {score}")
                    print(" --")

                results[field_key[0]] = score
        if verbose:
            print("===============================")

        return results

    def _score_list(self, predicted: List[str], truth: List[str]) -> float:
        """Scores a predicted list of strings against a ground truth list of strings"""
        # lower case all strings
        predictions = [prediction.lower() for prediction in predicted]
        truth_list = [truth_val.lower() for truth_val in truth if truth_val != "NULL"]

        # compare two lists of strings ignoring order
        pred_set = set(predictions)
        truth_set = set(truth_list)
        return 1.0 if pred_set == truth_set else 0.0

    def _score_meteor(self, predicted: str, truth: str) -> float:
        """Scores a predicted string against a ground truth string"""
        prediction_list = predicted.split(" ")
        truth_list = truth.split(" ")
        return single_meteor_score(prediction_list, truth_list)  # type: ignore

    def _score_string(self, predicted: str, truth: str) -> float:
        """Scores a predicted string against a ground truth string"""
        return 1.0 if predicted.lower() == truth.lower() else 0.0
