#!/usr/bin/env python3
"""Test script to verify all imports work correctly."""

import sys


def _test_import(module_name):
    try:
        __import__(module_name)
        print(f"✓ {module_name}")
        return True
    except Exception as e:
        print(f"✗ {module_name}: {e}")
        return False


def test_all_imports():
    modules = [
        "lovdata_rag.config",
        "lovdata_rag.logging_utils",
        "lovdata_rag.text_utils",
        "lovdata_rag.data_pipeline",
        "lovdata_rag.chunking",
        "lovdata_rag.embeddings",
        "lovdata_rag.bootstrap",
        "lovdata_rag.ft",
        "lovdata_rag.update",
        "api.bm25",
        "api.rag_pipeline",
        "api.api_server",
        "app",
        "scripts.eval_rag",
    ]

    print(f"Python version: {sys.version}")
    print("Testing imports...")
    print("-" * 50)

    results = []
    for mod in modules:
        results.append(_test_import(mod))

    print("-" * 50)
    print(f"Results: {sum(results)}/{len(results)} passed")
    assert all(results), f"Failed imports: {[modules[i] for i, r in enumerate(results) if not r]}"


if __name__ == "__main__":
    test_all_imports()
