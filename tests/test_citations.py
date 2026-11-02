import pytest

from backend.citations import citations_for_answer


def test_citations_only_return_exact_retrieved_source_pages_and_dedupe():
    contexts = [
        {"id": "chunk-a", "source": "annual, report.pdf", "page": 4},
        {"id": "chunk-b", "source": "annual, report.pdf", "page": 4},
        {"id": "chunk-c", "source": "guide.pdf", "page": 9},
    ]
    answer = "First claim [Source: annual, report.pdf, p. 4]. Second [Source: guide.pdf, p. 9]."
    assert citations_for_answer(answer, contexts) == [
        {"source": "annual, report.pdf", "page": 4, "chunk_id": "chunk-a"},
        {"source": "guide.pdf", "page": 9, "chunk_id": "chunk-c"},
    ]


@pytest.mark.parametrize("answer", [
    "Claim [Source: other.pdf, p. 1].",
    "Claim [Source: guide.pdf, p. 8].",
    "Claim [Source: guide.pdf page 1].",
    "An uncited factual claim.",
])
def test_rejects_unsupported_malformed_or_missing_citations(answer):
    with pytest.raises(ValueError):
        citations_for_answer(answer, [{"id": "a", "source": "guide.pdf", "page": 1}])


def test_allows_uncited_abstention():
    assert citations_for_answer("I cannot answer from the supplied context.", [{"source": "guide.pdf", "page": 1}]) == []
