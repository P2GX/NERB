import pytest

from nerb.benchmarking import benchmark_corpus, benchmark_doc
from nerb.models import DocId
from fixtures import extracted, ground_truth, doc1_extracted, doc1_truth


def test_benchmark():
    corpus_benchmarking_result = benchmark_corpus(extracted(), ground_truth())

    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc1")].true_positives == 2
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc1")].false_positives == 1
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc1")].false_negatives == 1

    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc2")].true_positives == 1
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc2")].false_positives == 0
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc2")].false_negatives == 0

    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc3")].true_positives == 0
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc3")].false_positives == 1
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc3")].false_negatives == 1

    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc4")].true_positives == 0
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc4")].false_positives == 1
    assert corpus_benchmarking_result.doc_benchmarks[DocId("doc4")].false_negatives == 0

    assert corpus_benchmarking_result.av_precision == pytest.approx(0.4166667)
    assert corpus_benchmarking_result.av_recall == pytest.approx(0.4166667)
    assert corpus_benchmarking_result.av_f1 == pytest.approx(0.4166667)


def test_benchmark_doc():
    doc_benchmarking_result = benchmark_doc(doc1_extracted(), doc1_truth())

    assert doc_benchmarking_result.true_positives == 2
    assert doc_benchmarking_result.false_positives == 1
    assert doc_benchmarking_result.false_negatives == 1
