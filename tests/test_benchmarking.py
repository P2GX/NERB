from pathlib import Path

import pytest

from src.nerb.benchmark import benchmark, load_dict_from_json  # adjust to where benchmark.py lives
from src.nerb.models import HpoTerm

ASSETS = Path(__file__).parent / "assets"
GROUND_TRUTH_PATH = ASSETS / "ground_truth.json"
EXTRACTED_PATH = ASSETS / "extracted.json"


@pytest.fixture
def ground_truth() -> dict[str, list[HpoTerm]]:
    return load_dict_from_json(GROUND_TRUTH_PATH)


@pytest.fixture
def extracted() -> dict[str, list[HpoTerm]]:
    return load_dict_from_json(EXTRACTED_PATH)


def test_load_dict_from_json(ground_truth):
    assert set(ground_truth) == {"doc1", "doc2", "doc3"}
    assert ground_truth["doc2"] == [HpoTerm("HP:0000252")]


def test_benchmark_on_assets(extracted, ground_truth):
    # doc1: 2 correct, 1 wrong (HP:0002650), 1 missed (HP:0001263)
    # doc2: 1 correct (duplicate prediction counted once)
    # doc3: not extracted -> 1 missed
    # doc4: not in ground truth -> 1 wrong
    result = benchmark(extracted, ground_truth)

    assert result.true_positives == 3
    assert result.false_positives == 2
    assert result.false_negatives == 2
    assert result.precision == pytest.approx(0.6)
    assert result.recall == pytest.approx(0.6)
    assert result.f1 == pytest.approx(0.6)


def test_benchmark_perfect_match(ground_truth):
    result = benchmark(ground_truth, ground_truth)

    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.precision == result.recall == result.f1 == 1.0


def test_benchmark_empty_inputs():
    result = benchmark({}, {})

    assert result.precision == result.recall == result.f1 == 0.0