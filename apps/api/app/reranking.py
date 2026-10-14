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
    "Given a repository question, retrieve relevant code passages "
    "that answer the question"
)
RERANKER_MAX_LENGTH = 32768
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
    model: CrossEncoder = CrossEncoder(
        RERANKER_MODEL,
        revision=RERANKER_REVISION,
        cache_folder=str(Path(__file__).resolve().parents[3] / ".model-cache"),
        trust_remote_code=False,
        device=device,
        model_kwargs={
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        },
        # Relevance scoring uses one forward pass, so a KV cache wastes VRAM.
        config_kwargs={"use_cache": False},
        max_length=RERANKER_MAX_LENGTH,
        prompts={"query": RERANKER_INSTRUCTION},
        default_prompt_name="query",
        activation_fn=torch.nn.Identity(),
    )
    return model


def score_passages(question: str, passages: list[str]) -> list[float]:
    """Return finite yes/no logit differences; errors surface without RRF fallback."""
    if not passages:
        return []
    # A cancelled async caller can leave inference running in its worker thread.
    # Serialize requests so a later request cannot overlap that GPU operation.
    with _SCORING_LOCK:
        model = get_reranker()
        pairs = [(question, passage) for passage in passages]
        processing = {"text": {"truncation": False}}
        for pair in pairs:
            features = model.preprocess(
                [pair],
                prompt=RERANKER_INSTRUCTION,
                processing_kwargs=processing,
            )
            if features["input_ids"].shape[-1] > RERANKER_MAX_LENGTH:
                raise ValueError(
                    "Reranker input exceeds the complete-window token limit."
                )
        raw = model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            processing_kwargs=processing,
        )
    scores = [float(value) for value in raw]
    if len(scores) != len(passages) or not all(map(math.isfinite, scores)):
        raise ValueError("Reranker returned invalid relevance scores.")
    return scores
