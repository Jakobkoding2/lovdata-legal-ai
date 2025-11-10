#!/usr/bin/env python3

from __future__ import annotations

import argparse

from openai import OpenAI

from lovdata_rag.ft import resolve_active_model


def parse_args():
    parser = argparse.ArgumentParser(description="Test fine-tuned model with a sample question")
    parser.add_argument("question", help="Question to send to the model")
    parser.add_argument("--default-model", default="gpt-5-mini", help="Fallback base model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = resolve_active_model(args.default_model)
    client = OpenAI()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "text", "text": "Du er en norsk jurist."}]},
            {"role": "user", "content": [{"type": "text", "text": args.question}]},
        ],
        temperature=0.2,
        max_output_tokens=400,
    )
    print(response.output_text)


if __name__ == "__main__":
    main()
