"""Hugging Face Spaces entrypoint for the Gradio demo."""

from app.gradio_app import build_app


app = build_app()
