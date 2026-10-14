"""Optional local relevance scoring with Qwen's dedicated, pinned reranker."""

from __future__ import annotations

import math
import threading
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
RERANKER_REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
RERANKER_INSTRUCTION = (
    "Given a repository question, retrieve relevant code passages that answer the question"
)
RERANKER_MAX_LENGTH = 8192
RERANKER_BATCH_SIZE = 4
_SCORING_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load official weights once; callers serialize loading and scoring off-loop."""
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as error:
        raise RuntimeError(
            "Qwen reranking requires the optional dependencies: "
            "run python scripts/uv_run.py --extra rerank <command>."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return CrossEncoder(
        RERANKER_MODEL,
        revision=RERANKER_REVISION,
        cache_folder=str(Path(__file__).resolve().parents[3] / ".model-cache"),
        trust_remote_code=False,
        device=device,
        model_kwargs={"torch_dtype": torch.float16 if device == "cuda" else torch.float32},
        max_length=RERANKER_MAX_LENGTH,
        prompts={"query": RERANKER_INSTRUCTION},
        default_prompt_name="query",
        activation_fn=torch.nn.Identity(),
    )


def score_passages(question: str, passages: list[str]) -> list[float]:
    """Return finite yes/no logit differences; errors surface without RRF fallback."""
    if not passages:
        return []
    # A cancelled async caller can leave inference running in its worker thread.
    # Serialize requests so a later request cannot overlap that GPU operation.
    with _SCORING_LOCK:
        raw = get_reranker().predict(
            [(question, passage) for passage in passages],
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    scores = [float(value) for value in raw]
    if len(scores) != len(passages) or not all(map(math.isfinite, scores)):
        raise ValueError("Reranker returned invalid relevance scores.")
    return scores
