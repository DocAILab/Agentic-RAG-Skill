from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from framework import load_framework_config

PROJECT_ROOT = Path(__file__).parents[1]
DEMO_ROOT = PROJECT_ROOT / "experiments" / "hotpotqa" / "data" / "demo"
OPTIMIZED_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "hotpotqa"
    / "configs"
    / "sim_rag_optimized.example.yaml"
)


def _load_jsonl(path: Path) -> list[dict]:
    """读取 demo JSONL 文件并返回对象列表。"""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _sha256(path: Path) -> str:
    """计算测试文件的 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_hotpotqa_demo_has_expected_stratified_counts() -> None:
    """验证 demo 的问题类型与答案类型配额符合固定设计。"""
    tests = _load_jsonl(DEMO_ROOT / "test.jsonl")
    strata = Counter((record["type"], record["answer_type"]) for record in tests)

    assert len(tests) == 100
    assert strata == {
        ("bridge", "span"): 50,
        ("comparison", "span"): 30,
        ("comparison", "yes"): 10,
        ("comparison", "no"): 10,
    }
    assert all(record["level"] == "hard" for record in tests)


def test_hotpotqa_demo_references_existing_corpus_and_sentences() -> None:
    """验证候选文档、相关文档和 supporting facts 均指向有效语料。"""
    corpus = {
        record["id"]: record for record in _load_jsonl(DEMO_ROOT / "corpus.jsonl")
    }
    tests = _load_jsonl(DEMO_ROOT / "test.jsonl")

    assert len(corpus) == 2000
    for example in tests:
        candidates = set(example["candidate_document_ids"])
        relevant = set(example["relevant_document_ids"])
        assert len(candidates) == 10
        assert candidates <= corpus.keys()
        assert relevant <= candidates
        assert len(relevant) >= 2
        for fact in example["supporting_facts"]:
            document = corpus[fact["document_id"]]
            assert 0 <= fact["sentence_id"] < len(document["sentences"])


def test_hotpotqa_demo_manifest_hashes_match_tracked_files() -> None:
    """验证已提交 demo 文件与 manifest 中的哈希一致。"""
    manifest = json.loads((DEMO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    for filename in ("corpus.jsonl", "test.jsonl"):
        assert _sha256(DEMO_ROOT / filename) == manifest["files"][filename]["sha256"]


def test_optimized_sim_rag_config_is_ready_for_controlled_rerun() -> None:
    config = load_framework_config(OPTIMIZED_CONFIG)

    assert config.demo is not None
    assert config.demo.max_examples == 20
    assert config.demo.candidate_documents_only is True
    assert config.demo.request["top_k"] == 3
    assert config.demo.request["max_iterations"] == 3
    assert config.demo.request["max_tokens"] == 256
    assert config.demo.request["critic_max_tokens"] == 4096
    assert config.demo.request["constraints"]["agentic_skill"] == "agentic-sim-rag"
    assert config.embedding is None
    assert config.executor.base_url == "https://api.deepseek.com"
    assert config.executor.api_key_env == "DEEPSEEK_API_KEY"


def test_hotpotqa_demo_rebuild_is_reproducible_when_raw_data_exists(
    tmp_path,
) -> None:
    """在本地原始分片存在时验证固定种子能够完全重建 demo。"""
    pytest.importorskip("pyarrow")
    from experiments.hotpotqa.scripts.build_demo import (
        DEFAULT_SEED,
        DEFAULT_SOURCE,
        build_demo,
    )

    if not DEFAULT_SOURCE.is_file():
        pytest.skip("Raw HotpotQA validation shard is not included in the repository")
    manifest = json.loads((DEMO_ROOT / "manifest.json").read_text(encoding="utf-8"))
    rebuilt = build_demo(DEFAULT_SOURCE, tmp_path, seed=DEFAULT_SEED)

    assert rebuilt["counts"] == manifest["counts"]
    assert rebuilt["files"]["corpus.jsonl"]["sha256"] == manifest["files"][
        "corpus.jsonl"
    ]["sha256"]
    assert rebuilt["files"]["test.jsonl"]["sha256"] == manifest["files"][
        "test.jsonl"
    ]["sha256"]
