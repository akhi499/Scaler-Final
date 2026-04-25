"""Hugging Face Spaces entrypoint for the Gradio demo."""

from app.gradio_app import build_app


app = build_app()


if __name__ == "__main__":
    # Explicit launch for local runs and deterministic Space startup behavior.
    app.launch(server_name="0.0.0.0", server_port=7860)
