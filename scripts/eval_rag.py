#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List

from openai import OpenAI, OpenAIError

from api.rag_pipeline import CodexRAGPipeline
from lovdata_rag import BASE_DIR
from lovdata_rag.bootstrap import ensure_assets_ready
from lovdata_rag.ft import resolve_active_model
from lovdata_rag.logging_utils import get_logger

logger = get_logger("lovdata_rag.eval")

EVAL_PATH = Path("eval/evalset.jsonl")
REPORT_JSON = Path("eval/report.json")
REPORT_HTML = Path("eval/report.html")


def load_evalset(limit: int | None = None) -> List[Dict]:
    data: List[Dict] = []
    with EVAL_PATH.open("r", encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            data.append(json.loads(line))
            if limit and idx + 1 >= limit:
                break
    return data


def compute_retrieval_metrics(results: List[Dict]) -> Dict[str, float]:
    hits = [1 if item["retrieval_hit"] else 0 for item in results]
    return {"retrieval@5": statistics.mean(hits) if hits else 0.0}


def extract_citations(answer: str) -> List[str]:
    citations: List[str] = []
    for token in answer.split("\n"):
        token = token.strip()
        if "\u00a7" in token:
            citations.append(token)
    return citations


def compute_citation_scores(results: List[Dict]) -> Dict[str, float]:
    precisions = []
    recalls = []
    f1s = []
    for item in results:
        predicted = set(item.get("citations", []))
        expected = set(item["expected_sections"])
        if not predicted and not expected:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        true_positive = sum(
            1 for p in predicted for e in expected if e.lower() in p.lower() or p.lower() in e.lower()
        )
        precision = true_positive / len(predicted) if predicted else 0.0
        recall = true_positive / len(expected) if expected else 0.0
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return {
        "citation_precision": statistics.mean(precisions) if precisions else 0.0,
        "citation_recall": statistics.mean(recalls) if recalls else 0.0,
        "citation_f1": statistics.mean(f1s) if f1s else 0.0,
    }


def evaluate_answers(results: List[Dict]) -> Dict[str, float]:
    matches = [1 if item.get("exact_match") else 0 for item in results if "exact_match" in item]
    return {"answer_exact_match": statistics.mean(matches) if matches else 0.0}


def generate_answer(client: OpenAI, model_name: str, question: str, context: str) -> str:
    response = client.responses.create(
        model=model_name,
        input=[
            {
                "role": "system",
                "content": [{"type": "text", "text": "Svar som norsk jurist og oppgi kilder."}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Kontekst:\n{context}\n\nSpørsmål: {question}",
                    }
                ],
            },
        ],
        temperature=0.2,
        max_output_tokens=600,
    )
    return getattr(response, "output_text", "").strip()


def run_eval(limit: int | None = None) -> None:
    ensure_assets_ready()
    pipeline = CodexRAGPipeline(BASE_DIR)
    evalset = load_evalset(limit=limit)
    client = None
    try:
        client = OpenAI()
    except Exception as exc:  # pragma: no cover
        logger.warning("OpenAI client unavailable: %s", exc)

    active_model = resolve_active_model("gpt-5-mini")
    baseline_model = "gpt-4o-mini"

    results: List[Dict] = []
    for example in evalset:
        question = example["question"]
        expected_sections = example["expected_sections"]
        retrieved = pipeline.search(question, top_k=5)
        retrieved_sections = [
            f"{item.doc_title} {item.section_num or ''}".strip() for item in retrieved
        ]
        hit = any(
            any(exp.lower() in section.lower() for section in retrieved_sections if section)
            for exp in expected_sections
        )
        entry = {
            "question": question,
            "expected_sections": expected_sections,
            "retrieved_sections": retrieved_sections,
            "retrieval_hit": hit,
        }

        if client:
            contexts = pipeline.top_contexts(question)
            context_text = "\n\n".join(f"{ctx.doc_title} {ctx.section_num}:\n{ctx.text}" for ctx in contexts)
            try:
                active_answer = generate_answer(client, active_model, question, context_text)
                baseline_answer = generate_answer(client, baseline_model, question, context_text)
                entry["active_answer"] = active_answer
                entry["baseline_answer"] = baseline_answer
                entry["citations"] = extract_citations(active_answer)
                entry["exact_match"] = any(
                    ref.lower() in active_answer.lower() for ref in example.get("reference_answer", "").split()
                )
            except OpenAIError as exc:  # pragma: no cover
                logger.warning("Skipping answer eval due to API error: %s", exc)

        results.append(entry)

    metrics = {}
    metrics.update(compute_retrieval_metrics(results))
    metrics.update(compute_citation_scores(results))
    metrics.update(evaluate_answers(results))

    REPORT_JSON.write_text(json.dumps({"metrics": metrics, "examples": results}, indent=2), encoding="utf-8")
    REPORT_HTML.write_text(
        "<html><body><h1>Lovdata RAG Evaluation</h1>"
        + "".join(f"<p><strong>{idx+1}. {item['question']}</strong><br>{item.get('active_answer','N/A')}</p>" for idx, item in enumerate(results))
        + f"<h2>Metrics</h2><pre>{json.dumps(metrics, indent=2)}</pre></body></html>",
        encoding="utf-8",
    )
    logger.info("Evaluation complete. Metrics saved to %s", REPORT_JSON)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval questions")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(limit=args.limit)
