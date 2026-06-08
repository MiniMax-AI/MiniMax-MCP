import importlib
import sys

from mcp.types import TextContent


def load_server(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://api.example.com")
    sys.modules.pop("minimax_mcp.server", None)
    return importlib.import_module("minimax_mcp.server")


def test_text_to_audio_returns_text_content_for_empty_text(monkeypatch):
    server = load_server(monkeypatch)

    result = server.text_to_audio(text="")

    assert isinstance(result, TextContent)
    assert result.type == "text"
    assert result.text == "Failed to generate audio: Text is required."
