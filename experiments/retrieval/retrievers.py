"""把现有 Component Skill 包装为独立检索评测器。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from framework.models import EmbeddingClient, SentenceTransformerEmbeddingClient
from framework.spec import load_runtime_callable, load_spec

from .baselines import run_original_bm25, run_title_weighted_bm25
from .schema import RetrievalExample

COMPONENTS = Path(__file__).parents[2] / "framework" / "skills" / "components"
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"
DEFAULT_BM25_VARIANT = "B3"
DEFAULT_BM25_K1 = 1.2
DEFAULT_BM25_B = 0.5
DEFAULT_BM25_TITLE_B = 0.75
DEFAULT_BM25_TITLE_BOOST = 3.0
DEFAULT_BGE_BATCH_SIZE = 8


@dataclass(slots=True)
class ComponentRetriever:
    name: str
    component: Any
    context: Any
    request_options: dict[str, Any] = field(default_factory=dict)
    document_fields: tuple[str, ...] = ("title", "text")

    def retrieve(self, example: RetrievalExample, *, top_k: int) -> list[dict]:
        request = example.to_request(top_k=top_k)
        if self.document_fields != ("title", "text"):
            request["documents"] = _project_documents(
                request["documents"], self.document_fields
            )
        request.update(self.request_options)
        result = self.component(request, self.context)
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
    variant: str | None = None,
    k1: float = DEFAULT_BM25_K1,
    b: float = DEFAULT_BM25_B,
    title_b: float | None = DEFAULT_BM25_TITLE_B,
    title_boost: float = DEFAULT_BM25_TITLE_BOOST,
    model: str = "BAAI/bge-large-en-v1.5",
    device: str | None = None,
    batch_size: int = DEFAULT_BGE_BATCH_SIZE,
    embedding_model: EmbeddingClient | None = None,
) -> ComponentRetriever:
    normalized = name.strip().lower()
    if normalized not in {"bm25", "vector"}:
        raise ValueError(f"Unsupported retriever: {name}")
    resolved_variant = resolve_variant(normalized, variant)
    if normalized == "bm25":
        component = _bm25_component(resolved_variant)
        options = {
            "k1": k1,
            "b": b,
        }
        if resolved_variant != "B0":
            options["title_boost"] = title_boost
        if resolved_variant in {"B2", "B3"}:
            options["title_b"] = b if title_b is None else title_b
        return ComponentRetriever(resolved_variant, component, object(), options)
    package = "component-vector-retriever"
    component = load_runtime_callable(load_spec(COMPONENTS / package))
    embedding = embedding_model or SentenceTransformerEmbeddingClient(
        model=model,
        device=device,
        batch_size=batch_size,
        normalize_embeddings=True,
    )
    if embedding_model is None:
        embedding.load()
    options = {"query_instruction": ""} if resolved_variant == "V0" else {}
    fields = ("text",) if resolved_variant in {"V0", "V1"} else ("title", "text")
    return ComponentRetriever(
        resolved_variant,
        component,
        EmbeddingContext(embedding),
        options,
        fields,
    )


def resolve_variant(retriever: str, variant: str | None) -> str:
    family = retriever.strip().lower()
    default = {"bm25": DEFAULT_BM25_VARIANT, "vector": "V2"}
    allowed = {"bm25": {"B0", "B1", "B2", "B3"}, "vector": {"V0", "V1", "V2"}}
    if family not in default:
        raise ValueError(f"Unsupported retriever: {retriever}")
    resolved = variant.upper() if variant else default[family]
    if resolved not in allowed[family]:
        raise ValueError(f"Variant {resolved} does not belong to {family}")
    return resolved


def _bm25_component(variant: str):
    if variant == "B0":
        return run_original_bm25
    if variant == "B1":
        return run_title_weighted_bm25
    package = "component-bm25-retriever"
    return load_runtime_callable(load_spec(COMPONENTS / package))


def _project_documents(documents, fields):
    projected = []
    for document in documents:
        item = dict(document)
        for field_name in ("title", "text"):
            if field_name not in fields:
                item[field_name] = ""
        item.pop("embedding", None)
        projected.append(item)
    return projected
