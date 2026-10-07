"""Regression tests for generated voices in list_voices responses."""
import importlib
from unittest.mock import Mock

import pytest
import anyio


@pytest.fixture
def server(monkeypatch):
    import dotenv
    import requests

    # Prevent reading .env files or using real credentials during import.
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-test-placeholder")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://api.example.invalid")

    def fail_network(*args, **kwargs):
        raise AssertionError("Network access is forbidden in this test")

    monkeypatch.setattr(requests.sessions.Session, "request", fail_network)
    return importlib.import_module("minimax_mcp.server")


def call_list_voices(server, voice_type="all"):
    # Official SDK in-memory transport; no stdio, sockets or listener.
    from mcp import Client

    async def invoke():
        async with Client(server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool("list_voices", {"voice_type": voice_type})
            assert result.is_error is False
            return result.content[0]

    return anyio.run(invoke)


@pytest.mark.parametrize("voice_type", ["all", "voice_generation"])
def test_list_voices_includes_generated_voices(server, monkeypatch, voice_type):
    fake_post = Mock(return_value={
        "system_voice": [],
        "voice_cloning": [],
        "voice_generation": [
            {"voice_id": "designed-1", "description": [], "created_time": "2026-10-06"},
            {"voice_id": "designed-2", "description": [], "created_time": "2026-10-06"},
        ],
    })
    monkeypatch.setattr(server.api_client, "post", fake_post)

    result = call_list_voices(server, voice_type)

    fake_post.assert_called_once_with("/v1/get_voice", json={"voice_type": voice_type})
    assert result.type == "text"
    assert "Voice Generation Voices:" in result.text
    assert "designed-1" in result.text
    assert "designed-2" in result.text


def test_list_voices_keeps_all_categories(server, monkeypatch):
    fake_post = Mock(return_value={
        "system_voice": [{"voice_id": "system-1", "voice_name": "System"}],
        "voice_cloning": [{"voice_id": "clone-1"}],
        "voice_generation": [{"voice_id": "designed-1"}],
    })
    monkeypatch.setattr(server.api_client, "post", fake_post)

    text = call_list_voices(server).text

    for voice_id in ("system-1", "clone-1", "designed-1"):
        assert voice_id in text


@pytest.mark.parametrize("response", [{}, {"voice_generation": None}, {"voice_generation": []}])
def test_list_voices_preserves_legacy_empty_response(server, monkeypatch, response):
    monkeypatch.setattr(server.api_client, "post", Mock(return_value=response))

    assert call_list_voices(server).text == "Success. System Voices: [], Voice Cloning Voices: []"


def test_list_voices_explicit_empty_generated_category(server, monkeypatch):
    monkeypatch.setattr(server.api_client, "post", Mock(return_value={"voice_generation": None}))

    assert call_list_voices(server, "voice_generation").text.endswith("Voice Generation Voices: []")
