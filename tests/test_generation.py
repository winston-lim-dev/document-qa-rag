from __future__ import annotations

import document_qa.generation as generation


def test_ollama_generator_uses_fixed_model_and_returns_content(monkeypatch) -> None:
    captured = {}

    def fake_chat(*, model, messages):
        captured["model"] = model
        captured["messages"] = messages
        return {"message": {"content": "Generated answer"}}

    monkeypatch.setattr(generation, "chat", fake_chat)

    answer = generation.OllamaGenerator()("Grounded prompt")

    assert answer == "Generated answer"
    assert captured == {
        "model": "llama3.2:3b",
        "messages": [{"role": "user", "content": "Grounded prompt"}],
    }
