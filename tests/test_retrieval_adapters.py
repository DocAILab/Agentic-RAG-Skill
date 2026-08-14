from __future__ import annotations

import json

import pytest

from experiments.retrieval.adapters import (
    AdapterError,
    adapt_hotpotqa,
    adapt_triviaqa,
    adapt_two_wiki,
)


def test_hotpotqa_adapts_column_oriented_context_and_gold_titles() -> None:
    example = adapt_hotpotqa(
        {
            "id": "hp-1",
            "question": "Who won?",
            "context": {
                "title": ["Alpha", "Beta"],
                "sentences": [["Alpha text."], ["Beta one.", "Beta two."]],
            },
            "supporting_facts": {"title": ["Alpha", "Beta"], "sent_id": [0, 1]},
        }
    )

    assert example.id == "hp-1"
    assert [document.id for document in example.documents] == ["Alpha", "Beta"]
    assert example.documents[1].text == "Beta one. Beta two."
    assert example.relevant_document_ids == ("Alpha", "Beta")
    assert example.label_type == "supporting_facts"


def test_two_wiki_adapts_record_oriented_context() -> None:
    example = adapt_two_wiki(
        {
            "_id": "wiki-1",
            "question": "Where?",
            "context": [
                {"title": "First", "content": ["One."]},
                {"title": "Second", "content": ["Two."]},
            ],
            "supporting_facts": [{"title": "Second", "sent_id": 0}],
        }
    )

    assert example.id == "wiki-1"
    assert example.relevant_document_ids == ("Second",)
    assert example.documents[0].text == "One."


def test_two_wiki_adapts_json_strings_from_official_parquet() -> None:
    example = adapt_two_wiki(
        {
            "_id": "wiki-json",
            "question": "Where?",
            "context": json.dumps([["First", ["One."]], ["Second", ["Two."]]]),
            "supporting_facts": json.dumps([["Second", 0]]),
        }
    )

    assert example.relevant_document_ids == ("Second",)
    assert [document.text for document in example.documents] == ["One.", "Two."]


def test_triviaqa_builds_weak_labels_from_answer_aliases() -> None:
    example = adapt_triviaqa(
        {
            "question_id": "trivia-1",
            "question": "What is the capital?",
            "entity_pages": [
                {"title": "France", "wiki_context": "Paris is the capital of France."}
            ],
            "search_results": {
                "title": ["Unrelated"],
                "search_context": ["London is in the United Kingdom."],
            },
            "answer": {"aliases": ["PARIS"], "value": "Paris"},
        }
    )

    assert example.label_type == "weak_answer_alias"
    assert example.relevant_document_ids == ("entity:0",)
    assert example.metadata["weak_labels"] is True


def test_unlabelled_examples_remain_unlabelled() -> None:
    example = adapt_hotpotqa(
        {
            "id": "test-1",
            "question": "Hidden answer?",
            "context": {"title": ["Candidate"], "sentences": [["Text"]]},
            "supporting_facts": {"title": [], "sent_id": []},
        }
    )

    assert example.relevant_document_ids == ()
    assert example.label_type is None
    assert example.has_labels is False


def test_missing_fields_report_sample_id() -> None:
    with pytest.raises(AdapterError, match="broken.*question") as error:
        adapt_two_wiki({"_id": "broken", "context": []})

    assert error.value.sample_id == "broken"
