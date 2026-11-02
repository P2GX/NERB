import json
from pathlib import Path
from typing import Any

from nerb.models import CorpusAnnotations, DocId, DocAnnotation, HpoId


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def load_annotations(json_path: Path) -> CorpusAnnotations:
    """Load a CorpusAnnotations from a JSON file of the form
    {"doc_id": ["HP:0001250", ...], ...}.
    """
    with Path(json_path).open(encoding="utf-8") as f:
        data: Any = json.load(f)

    doc_to_annotations: dict[DocId, DocAnnotation] = {}
    for doc_id_str, hpo_id_strs in data.items():
        doc_id = DocId(doc_id_str)
        hpo_ids: list[HpoId] = [HpoId(hpo_id_str) for hpo_id_str in hpo_id_strs]
        doc_to_annotations[doc_id] = DocAnnotation(hpo_ids)

    return CorpusAnnotations(doc_to_annotations)
