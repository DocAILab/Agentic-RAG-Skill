from __future__ import annotations

from experiments.retrieval.loading import DatasetItem
from experiments.retrieval.performance import benchmark_batch_sizes
from experiments.retrieval.run_bge_screening import build_parser, run_screening
from experiments.retrieval.schema import RetrievalDocument, RetrievalExample

BGE_INSTRUCTION = "Represent this sentence for searching relevant passages:"


class SemanticEmbeddingModel:
    def embed(self, texts):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        if text.startswith(BGE_INSTRUCTION):
            return [0.0, 1.0]
        if text == "orchid" or text.endswith("noise"):
            return [1.0, 0.0]
        return [0.0, 1.0]


def _items():
    example = RetrievalExample(
        id="one",
        query="orchid",
        documents=(
            RetrievalDocument("gold", "Gold", "semantic"),
            RetrievalDocument("noise", "Noise", "noise"),
        ),
        relevant_document_ids=("gold",),
        label_type="supporting_facts",
    )
    return [DatasetItem(0, "one", example=example)]


def test_bge_screening_reuses_model_and_freezes_best_improved_variant(tmp_path) -> None:
    model = SemanticEmbeddingModel()

    result = run_screening(
        {"hotpotqa": _items(), "2wiki": _items()},
        embedding_model=model,
        output_dir=tmp_path,
        model_name="fixture",
    )

    assert result["selected_variant"] == "V1"
    assert result["baseline_variant"] == "V0"
    assert len((tmp_path / "bge_variants.jsonl").read_text().splitlines()) == 3


def test_batch_benchmark_records_oom_without_failing_quality_run() -> None:
    class Client:
        def __init__(self, batch_size):
            self.batch_size = batch_size

        def embed(self, texts):
            if self.batch_size == 64:
                raise MemoryError("out of memory")
            return [[1.0, 0.0] for _ in texts]

    result = benchmark_batch_sizes(
        Client,
        [8, 64],
        ["a", "b"],
        warmup=0,
        repeats=1,
    )

    assert result[0]["status"] == "ok"
    assert result[1] == {
        "batch_size": 64,
        "status": "oom",
        "error": "out of memory",
    }


def test_bge_screening_defaults_to_measured_batch_size(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "--hotpot-manifest",
            str(tmp_path / "hotpot.json"),
            "--two-wiki-manifest",
            str(tmp_path / "2wiki.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert args.batch_size == 8
