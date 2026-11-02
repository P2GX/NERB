from dataclasses import dataclass


@dataclass(frozen=True)
class HpoId:
    """A wrapper around a HPO ID."""

    hpo_id: str


@dataclass(frozen=True)
class DocId:
    """The ID of a document."""

    doc_id: str


@dataclass(frozen=True)
class DocAnnotation:
    """
    This is what we expect to be annotated to a single document.
    """

    annotations: list[HpoId]


@dataclass(frozen=True)
class CorpusAnnotations:
    """
    - The expected output of our extraction software,
    - The format of our ground truth annotation,
    - The input to our benchmarking function
    """

    doc_to_annotations: dict[DocId, DocAnnotation]
