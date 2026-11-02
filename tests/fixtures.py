from nerb.models import CorpusAnnotations, DocId, DocAnnotation, HpoId


def doc1_truth() -> DocAnnotation:
    return DocAnnotation(
        [HpoId("HP:0001250"), HpoId("HP:0004322"), HpoId("HP:0001263")]
    )


def doc2_truth() -> DocAnnotation:
    return DocAnnotation([HpoId("HP:0000252")])


def doc3_truth() -> DocAnnotation:
    return DocAnnotation([HpoId("HP:0001629")])


def doc4_truth() -> DocAnnotation:
    return DocAnnotation([])


def ground_truth() -> CorpusAnnotations:
    doc_to_annotations = {
        DocId("doc1"): doc1_truth(),
        DocId("doc2"): doc2_truth(),
        DocId("doc3"): doc3_truth(),
        DocId("doc4"): doc4_truth(),
    }
    return CorpusAnnotations(doc_to_annotations)


def doc1_extracted() -> DocAnnotation:
    return DocAnnotation(
        [HpoId("HP:0001250"), HpoId("HP:0004322"), HpoId("HP:0002650")]
    )


def doc2_extracted() -> DocAnnotation:
    return DocAnnotation([HpoId("HP:0000252")])


def doc3_extracted() -> DocAnnotation:
    return DocAnnotation([HpoId("HP:0001631")])


def doc4_extracted() -> DocAnnotation:
    return DocAnnotation([HpoId("HP:0001631")])


def extracted() -> CorpusAnnotations:
    doc_to_annotations = {
        DocId("doc1"): doc1_extracted(),
        DocId("doc2"): doc2_extracted(),
        DocId("doc3"): doc3_extracted(),
        DocId("doc4"): doc4_extracted(),
    }
    return CorpusAnnotations(doc_to_annotations)
