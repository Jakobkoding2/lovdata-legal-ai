import os

import gradio as gr
import requests


def _resolve_api_url() -> str:
    """Return the backend base URL, defaulting to localhost for local dev."""

    base_url = os.getenv("API_URL") or os.getenv("BACKEND_URL")
    if not base_url:
        return "http://localhost:8000"
    return base_url.rstrip("/")


API_URL = _resolve_api_url()


def search_law(query: str, top_k: int):
    response = requests.post(f"{API_URL}/search", json={"query": query, "top_k": int(top_k)})
    response.raise_for_status()
    return response.json()


def ask_question(question: str):
    response = requests.post(f"{API_URL}/ask_law", json={"question": question})
    response.raise_for_status()
    return response.json()


with gr.Blocks() as demo:
    gr.Markdown("# Lovdata Legal AI")

    with gr.Tab("Semantic Search"):
        query_input = gr.Textbox(label="Search Query")
        top_k_slider = gr.Slider(1, 20, value=5, step=1, label="Number of Results")
        search_button = gr.Button("Search")
        search_output = gr.JSON(label="Results")
        search_button.click(search_law, inputs=[query_input, top_k_slider], outputs=search_output)

    with gr.Tab("Ask a Question"):
        question_input = gr.Textbox(label="Legal Question")
        ask_button = gr.Button("Ask")
        answer_output = gr.JSON(label="Answer")
        ask_button.click(ask_question, inputs=question_input, outputs=answer_output)


def main() -> None:
    demo.launch()


if __name__ == "__main__":
    main()
