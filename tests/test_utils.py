from pathlib import Path
from nerb.utils import load_annotations
from fixtures import ground_truth, extracted

ASSETS: Path = Path(__file__).parent / "assets"
GROUND_TRUTH_PATH: Path = ASSETS / "ground_truth.json"
EXTRACTED_PATH: Path = ASSETS / "extracted.json"


def test_load_annotations():
    assert ground_truth() == load_annotations(GROUND_TRUTH_PATH)
    assert extracted() == load_annotations(EXTRACTED_PATH)
