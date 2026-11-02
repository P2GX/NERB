from dataclasses import dataclass
from statistics import mean

from nerb.models import CorpusAnnotations, DocId, DocAnnotation, HpoId
from nerb.utils import _safe_div


@dataclass(frozen=True)
class DocBenchmarkingResult:
    """Contains benchmarking statistics of an extraction method for a single document."""

    precision: float
    recall: float
    f1: float
    true_positives: int
    false_positives: int
    false_negatives: int


@dataclass(frozen=True)
class CorpusBenchmarkingResult:
    """Contains all benchmarking statistics of an extraction method across an entire corpus."""

    doc_benchmarks: dict[DocId, DocBenchmarkingResult]
    av_precision: float
    av_recall: float
    av_f1: float


def benchmark_doc(
    extracted_annotations: DocAnnotation, ground_truth: DocAnnotation
) -> DocBenchmarkingResult:
    """Provides benchmarking statistics on a single document."""
    extracted_ids: set[HpoId] = set(extracted_annotations.annotations)
    expected_ids: set[HpoId] = set(ground_truth.annotations)

    tp: int = len(extracted_ids & expected_ids)
    fp: int = len(extracted_ids - expected_ids)
    fn: int = len(expected_ids - extracted_ids)

    precision: float = _safe_div(tp, tp + fp)
    recall: float = _safe_div(tp, tp + fn)
    f1: float = _safe_div(2 * precision * recall, precision + recall)

    return DocBenchmarkingResult(precision, recall, f1, tp, fp, fn)


def benchmark_corpus(
    extracted_annotations: CorpusAnnotations, ground_truth: CorpusAnnotations
) -> CorpusBenchmarkingResult:
    """Takes extracted annotations, compares it to the ground truth annotations, and returns an evaluation."""

    doc_ids: list[DocId] = list(ground_truth.doc_to_annotations.keys())
    doc_benchmarks: dict[DocId, DocBenchmarkingResult] = {}

    for doc_id in doc_ids:
        extracted: DocAnnotation = extracted_annotations.doc_to_annotations[doc_id]
        expected: DocAnnotation = ground_truth.doc_to_annotations[doc_id]
        doc_benchmark: DocBenchmarkingResult = benchmark_doc(extracted, expected)
        doc_benchmarks[doc_id] = doc_benchmark

    av_precision: float = mean(
        doc_benchmark.precision for doc_benchmark in doc_benchmarks.values()
    )
    av_recall: float = mean(
        doc_benchmark.recall for doc_benchmark in doc_benchmarks.values()
    )
    av_f1: float = mean(doc_benchmark.f1 for doc_benchmark in doc_benchmarks.values())

    return CorpusBenchmarkingResult(doc_benchmarks, av_precision, av_recall, av_f1)
