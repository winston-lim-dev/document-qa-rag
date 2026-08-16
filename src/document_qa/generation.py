"""Small production boundary for Ollama text generation."""

from __future__ import annotations

from typing import Any

from ollama import chat

DEFAULT_GENERATION_MODEL = "llama3.2:3b"


class OllamaGenerator:
    """Generate text with the application's fixed local Ollama model."""

    def __call__(self, prompt: str) -> str:
        response: Any = chat(
            model=DEFAULT_GENERATION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response["message"]["content"])
