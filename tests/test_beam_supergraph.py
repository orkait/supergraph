import json
from pathlib import Path


def test_beam_pair_chunking_builds_user_assistant_pairs():
    from benchmarks.framework.runners.beam import create_chunking

    messages = [
        {
            "batch_number": 1,
            "turns": [
                [
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                    {"role": "user", "content": "U2"},
                    {"role": "assistant", "content": "A2"},
                ]
            ],
        }
    ]

    chunks = create_chunking(messages, retrieval_method="pair_chunk")
    assert len(chunks) == 2
    assert "USER: U1" in chunks[0]["text"]
    assert "ASSISTANT: A1" in chunks[0]["text"]
    assert chunks[0]["metadata"]["pair_number"] == 1
    assert chunks[1]["metadata"]["pair_number"] == 2


def test_beam_turn_chunking_builds_whole_turn():
    from benchmarks.framework.runners.beam import create_chunking

    messages = [
        {
            "batch_number": 1,
            "turns": [
                [
                    {"role": "user", "content": "U1"},
                    {"role": "assistant", "content": "A1"},
                ]
            ],
        }
    ]

    chunks = create_chunking(messages, retrieval_method="turn_chunk")
    assert len(chunks) == 1
    assert "USER: U1" in chunks[0]["text"]
    assert "ASSISTANT: A1" in chunks[0]["text"]


def test_answer_payload_preserves_question_structure_and_adds_llm_response(tmp_path):
    from benchmarks.framework.runners.beam import build_answer_payload

    probing = {
        "abstention": [
            {"question": "Q1", "rubric": ["r1"]},
        ],
        "information_extraction": [
            {"question": "Q2", "rubric": ["r2"]},
        ],
    }
    answers = {
        ("abstention", 0): "A1",
        ("information_extraction", 0): "A2",
    }

    payload = build_answer_payload(probing, answers)
    assert payload["abstention"][0]["question"] == "Q1"
    assert payload["abstention"][0]["llm_response"] == "A1"
    assert payload["information_extraction"][0]["llm_response"] == "A2"
