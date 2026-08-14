"""把现有 Component Skill 包装为独立检索评测器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from framework.models import EmbeddingClient, SentenceTransformerEmbeddingClient
from framework.spec import load_runtime_callable, load_spec

from .schema import RetrievalExample

COMPONENTS = Path(__file__).parents[2] / "framework" / "skills" / "components"


@dataclass(slots=True)
class ComponentRetriever:
    name: str
    component: Any
    context: Any

    def retrieve(self, example: RetrievalExample, *, top_k: int) -> list[dict]:
        result = self.component(example.to_request(top_k=top_k), self.context)
        documents = result.get("documents")
        if not isinstance(documents, list):
            raise TypeError(f"{self.name} returned invalid documents")
        return documents


class EmbeddingContext:
    def __init__(self, model: EmbeddingClient):
        self.model = model

    def embed(self, texts):
        return self.model.embed(texts)


def build_retriever(
    name: str,
    *,
    model: str = "BAAI/bge-large-en-v1.5",
    device: str | None = None,
    batch_size: int = 32,
    embedding_model: EmbeddingClient | None = None,
) -> ComponentRetriever:
    normalized = name.strip().lower()
    if normalized not in {"bm25", "vector"}:
        raise ValueError(f"Unsupported retriever: {name}")
    package = f"component-{normalized}-retriever"
    component = load_runtime_callable(load_spec(COMPONENTS / package))
    if normalized == "bm25":
        return ComponentRetriever(normalized, component, object())
    embedding = embedding_model or SentenceTransformerEmbeddingClient(
        model=model,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    if embedding_model is None:
        embedding.load()
    return ComponentRetriever(normalized, component, EmbeddingContext(embedding))
