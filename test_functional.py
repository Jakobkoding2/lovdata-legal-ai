#!/usr/bin/env python3
"""Functional tests for lovdata-legal-ai components."""

import os
import sys
import tempfile
from pathlib import Path

# Set a temp base dir so we don't need real data
os.environ.setdefault("API_URL", "http://localhost:8000")

def test_text_utils():
    from lovdata_rag.text_utils import split_sentences, split_sections, split_ledd

    text = "Dette er setning én. Dette er setning to! Og nummer tre?"
    sentences = split_sentences(text)
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}"
    print("✓ split_sentences")

    sections = split_sections("§ 1 Første. § 2 Andre.")
    assert len(sections) == 2
    print("✓ split_sections")

    ledd = split_ledd("1. ledd Første. 2. ledd Andre.")
    assert len(ledd) == 2
    print("✓ split_ledd")


def test_bm25():
    from api.bm25 import BM25Okapi
    corpus = [["hello", "world"], ["hello", "there"]]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(["hello"])
    assert len(scores) == 2
    print("✓ BM25Okapi")


def test_config_paths():
    from lovdata_rag.config import BASE_DIR, DATA_DIR, MODELS_DIR
    assert BASE_DIR.exists()
    assert DATA_DIR.exists()
    assert MODELS_DIR.exists()
    print("✓ config paths")


def test_chunk_record():
    from api.rag_pipeline import ChunkRecord
    record = ChunkRecord(
        chunk_id="test_1",
        doc_id="lov_1",
        doc_title="Test Lov",
        section_num="1",
        section_title=None,
        group="law",
        text="Dette er en test.",
    )
    assert record.doc_title == "Test Lov"
    print("✓ ChunkRecord dataclass")


def test_api_models():
    from api.api_server import SearchRequest, QARequest, OverlapRequest
    sr = SearchRequest(query="test", top_k=5)
    assert sr.top_k == 5
    qa = QARequest(question="hva er loven?")
    assert qa.context_size == 5
    ov = OverlapRequest(text1="a", text2="b")
    assert ov.text1 == "a"
    print("✓ API request models")


def test_resolve_model():
    from lovdata_rag.ft import resolve_active_model
    # No env vars set
    assert resolve_active_model("gpt-4o-mini") == "gpt-4o-mini"
    os.environ["MODEL_NAME"] = "gpt-4o"
    assert resolve_active_model("gpt-4o-mini") == "gpt-4o"
    del os.environ["MODEL_NAME"]
    print("✓ resolve_active_model")


def test_compute_overlap():
    from api.rag_pipeline import compute_overlap
    # Without a real model, should return 0.0
    result = compute_overlap(None, "hello", "world")
    assert result == 0.0
    print("✓ compute_overlap fallback")


def test_app_functions():
    from app import _resolve_api_url
    assert _resolve_api_url() == "http://localhost:8000"
    os.environ["API_URL"] = "http://example.com"
    assert _resolve_api_url() == "http://example.com"
    del os.environ["API_URL"]
    print("✓ app _resolve_api_url")


def main():
    tests = [
        test_text_utils,
        test_bm25,
        test_config_paths,
        test_chunk_record,
        test_api_models,
        test_resolve_model,
        test_compute_overlap,
        test_app_functions,
    ]
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1

    print("-" * 50)
    print(f"Results: {passed}/{len(tests)} passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
