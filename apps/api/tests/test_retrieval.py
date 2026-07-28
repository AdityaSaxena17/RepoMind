from app.retrieval import RepoMindParentHybridRetriever


def test_parent_hybrid_retriever_fuses_vector_and_keyword_ranks() -> None:
  fused = RepoMindParentHybridRetriever._fuse_candidates(
    [
      {
        "id": "chunk_1",
        "file_id": "file_1",
        "path": "app.py",
        "start_line": 1,
        "end_line": 5,
        "source": "vector",
        "rank": 1,
      },
    ],
    [
      {
        "id": "chunk_1",
        "file_id": "file_1",
        "path": "app.py",
        "start_line": 1,
        "end_line": 5,
        "source": "keyword",
        "rank": 1,
      },
      {
        "id": "chunk_2",
        "file_id": "file_2",
        "path": "lib.py",
        "start_line": 1,
        "end_line": 5,
        "source": "keyword",
        "rank": 2,
      },
    ],
  )

  assert [item["id"] for item in fused] == ["chunk_1", "chunk_2"]
  assert fused[0]["sources"] == {"vector", "keyword"}
