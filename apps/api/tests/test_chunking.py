import pytest
from langchain_core.documents import Document

from app import worker
from app.worker import chunk_file


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_chunk_ranges_contain_the_matching_text(newline: str) -> None:
    content = (newline * 2).join(
        f"def unique_method_{index}():{newline}    return 'value_{index}'"
        for index in range(160)
    )
    documents = chunk_file("large.py", content)
    assert len(documents) > 1
    lines = content.splitlines()
    for document in documents:
        start = document.metadata["start_line"]
        end = document.metadata["end_line"]
        visible = newline.join(lines[start - 1 : end])
        assert document.page_content in visible
        assert 1 <= start <= end <= len(lines)


def test_chunk_text_missing_from_source_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        worker.RecursiveCharacterTextSplitter,
        "create_documents",
        lambda _self, _texts: [
            Document(page_content="missing", metadata={"start_index": -1})
        ],
    )
    with pytest.raises(ValueError, match="Chunk text does not occur in source"):
        chunk_file("sample.py", "actual")
