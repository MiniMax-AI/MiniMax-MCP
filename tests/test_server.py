import importlib
import os
import sys

import anyio


os.environ.setdefault("MINIMAX_API_KEY", "test-api-key")
os.environ.setdefault("MINIMAX_API_HOST", "https://api.example.invalid")

Client = importlib.import_module("mcp").Client
StdioServerParameters = importlib.import_module("mcp").StdioServerParameters
stdio_client = importlib.import_module("mcp.client.stdio").stdio_client
server = importlib.import_module("minimax_mcp.server")


EXPECTED_TOOLS = {
    "text_to_audio",
    "list_voices",
    "voice_clone",
    "play_audio",
    "generate_video",
    "query_video_generation",
    "text_to_image",
    "image_edit",
    "voice_design",
}


def test_initialize_and_list_tools():
    async def exercise_server():
        async with Client(server.mcp, raise_exceptions=True) as client:
            assert client.server_info.name == "Minimax"
            assert client.protocol_version

            result = await client.list_tools()
            assert {tool.name for tool in result.tools} == EXPECTED_TOOLS

    anyio.run(exercise_server)


def test_stdio_startup_and_list_tools():
    async def exercise_stdio_server():
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "minimax_mcp.server"],
            env={
                "MINIMAX_API_KEY": "test-api-key",
                "MINIMAX_API_HOST": "https://api.example.invalid",
            },
        )
        for mode in ("auto", "legacy"):
            async with Client(stdio_client(parameters), mode=mode) as client:
                result = await client.list_tools()
                assert {tool.name for tool in result.tools} == EXPECTED_TOOLS
                if mode == "legacy":
                    assert client.protocol_version != "2026-07-28"

    anyio.run(exercise_stdio_server)


def test_tool_call_uses_mocked_api(monkeypatch):
    def fake_post(path, **kwargs):
        assert path == "/v1/get_voice"
        return {
            "system_voice": [{"voice_name": "Test Voice", "voice_id": "test-voice"}],
            "voice_cloning": [],
        }

    monkeypatch.setattr(server.api_client, "post", fake_post)

    async def exercise_tool():
        async with Client(server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool("list_voices", {"voice_type": "system"})
            assert result.is_error is False
            assert "test-voice" in result.content[0].text

    anyio.run(exercise_tool)


def test_text_to_image_routes_h3_and_legacy(monkeypatch):
    calls = []

    def fake_post(path, **kwargs):
        calls.append((path, kwargs.get("json"), kwargs.get("timeout")))
        if path == "/v1/images/generations":
            return {"data": [{"url": "https://example.invalid/h3.jpg", "size": "1024x1024"}]}
        if path == "/v1/image_generation":
            return {"data": {"image_urls": ["https://example.invalid/legacy.jpg"]}}
        raise AssertionError(f"unexpected path {path}")

    monkeypatch.setattr(server.api_client, "post", fake_post)

    async def exercise_tool():
        async with Client(server.mcp, raise_exceptions=True) as client:
            h3 = await client.call_tool(
                "text_to_image",
                {"prompt": "an orange cat", "area": "1K"},
            )
            legacy = await client.call_tool(
                "text_to_image",
                {"prompt": "an orange cat", "model": "image-01"},
            )
            assert h3.is_error is False
            assert "https://example.invalid/h3.jpg" in h3.content[0].text
            assert legacy.is_error is False
            assert "https://example.invalid/legacy.jpg" in legacy.content[0].text

    anyio.run(exercise_tool)

    assert calls[0][0] == "/v1/images/generations"
    assert calls[0][1]["model"] == "MiniMax-H3-Image"
    assert calls[0][1]["area"] == "1K"
    assert calls[0][1]["aspect_ratio"] == "1:1"
    assert "prompt_optimizer" not in calls[0][1]
    assert calls[0][2] == 610
    assert calls[1][0] == "/v1/image_generation"
    assert calls[1][1] == {
        "model": "image-01",
        "prompt": "an orange cat",
        "aspect_ratio": "1:1",
        "n": 1,
        "prompt_optimizer": True,
    }
    assert calls[1][2] is None


def test_image_edit_posts_reference_images(monkeypatch):
    calls = []

    def fake_post(path, **kwargs):
        calls.append((path, kwargs.get("json"), kwargs.get("timeout")))
        return {"data": [{"url": "https://example.invalid/edit.jpg", "size": "1024x1024"}]}

    monkeypatch.setattr(server.api_client, "post", fake_post)

    async def exercise_tool():
        async with Client(server.mcp, raise_exceptions=True) as client:
            result = await client.call_tool(
                "image_edit",
                {
                    "prompt": "make it watercolor",
                    "images": ["https://example.invalid/ref.jpg"],
                    "area": "1K",
                },
            )
            assert result.is_error is False
            assert "https://example.invalid/edit.jpg" in result.content[0].text

    anyio.run(exercise_tool)

    assert calls[0][0] == "/v1/images/edits"
    assert calls[0][1]["images"] == [{"image_url": "https://example.invalid/ref.jpg"}]
    assert calls[0][1]["area"] == "1K"
    assert calls[0][2] == 610
