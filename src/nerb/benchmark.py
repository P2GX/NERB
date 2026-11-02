import json
from dataclasses import dataclass
from pathlib import Path

from src.nerb.models import HpoTerm
from src.nerb.utils import safe_div


@dataclass(frozen=True)
class BenchmarkingResult:
    """Micro-averaged precision, recall and F1 over the whole corpus. TODO MAKE THIS A NICER """

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int

def load_dict_from_json(json_path: Path) -> dict[str, list[HpoTerm]]:
    """Load {"item_id": ["HP:0001250", ...], ...} into {item_id: [HpoTerm, ...]}."""
    with Path(json_path).open(encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"{json_path}: expected a JSON object at the top level")

    result: dict[str, list[HpoTerm]] = {}
    for item_id, hpo_ids in data.items():
        if not isinstance(hpo_ids, list):
            raise ValueError(f"{json_path}: value for {item_id!r} must be a list of HPO IDs")
        result[str(item_id)] = [HpoTerm(hpo_id) for hpo_id in hpo_ids]
    return result

def benchmark(
    item_to_extracted_terms: dict[str, list[HpoTerm]],
    ground_truth: dict[str, list[HpoTerm]],
) -> BenchmarkingResult:
    """DOC_STRING TODO"""
    tp = fp = fn = 0

    # Documents missing from either side count as having no terms there:
    # missing predictions -> false negatives, extra documents -> false positives.
    for item_id in ground_truth.keys() | item_to_extracted_terms.keys():
        # Sets remove duplicate terms within a document.
        # Requires HpoTerm to be hashable and compare equal by HPO ID;
        # if it isn't, use {t.<id attribute> for t in ...} here instead.
        predicted = set(item_to_extracted_terms.get(item_id, []))
        expected = set(ground_truth.get(item_id, []))

        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return BenchmarkingResult(
        precision=precision,
        recall=recall,
        f1=f1,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )