"""Regression tests for issue #64: all requests.get() calls in server.py
must pass an explicit timeout to avoid hanging the MCP tool pipeline
indefinitely on a slow or unresponsive remote server.

These tests monkey-patch ``minimax_mcp.server.requests.get`` (the
module-level reference used inside the tool functions) and assert
that the timeout kwarg is forwarded. They will fail if any of the
call sites in server.py drops the ``timeout=DOWNLOAD_TIMEOUT`` kwarg.
"""
import io
from unittest.mock import MagicMock, patch

import pytest
import requests

import minimax_mcp.server as server
from minimax_mcp.const import DOWNLOAD_TIMEOUT


def test_download_timeout_constant_is_120():
    """The constant must be 120 seconds (issue #64 proposed value)."""
    assert DOWNLOAD_TIMEOUT == 120


def _make_fake_get_response(content: bytes = b"FAKE"):
    """Build a MagicMock that quacks like a requests.Response."""
    resp = MagicMock(spec=requests.Response)
    resp.status_code = 200
    resp.content = content
    resp.raw = io.BytesIO(content)
    resp.raise_for_status = MagicMock()
    # iter_content yields the full content in one chunk
    resp.iter_content = MagicMock(return_value=iter([content]))
    return resp


def test_voice_clone_is_url_uses_timeout(tmp_path):
    """voice_clone(is_url=True) downloads the file and must pass timeout."""
    fake_resp = _make_fake_get_response(b"audio-bytes")

    with patch.object(server, "resource_mode", "local"), \
         patch.object(server.requests, "get", return_value=fake_resp) as mock_get, \
         patch.object(server.api_client, "post") as mock_post, \
         patch("minimax_mcp.server.play") as mock_play:
        # First api_client.post returns the file-upload response with a file_id.
        # Second api_client.post returns the voice_clone response with demo_audio.
        mock_post.side_effect = [
            {"file": {"file_id": "fake_file_id"}},  # /v1/files/upload
            {"demo_audio": "https://example.com/demo.wav"},  # /v1/voice_clone
        ]
        result = server.voice_clone(
            voice_id="test_voice",
            file="https://example.com/source.mp3",
            text="hello",
            output_directory=str(tmp_path),
            is_url=True,
        )

    # The first requests.get call is the is_url download; verify its timeout.
    first_call = mock_get.call_args_list[0]
    assert first_call.kwargs.get("timeout") == DOWNLOAD_TIMEOUT, (
        f"voice_clone(is_url=True) requests.get missing timeout kwarg: {first_call}"
    )
    # The is_url path already used stream=True before the fix; preserve it.
    assert first_call.kwargs.get("stream") is True


def test_voice_clone_demo_audio_download_uses_timeout(tmp_path):
    """voice_clone() downloads the demo audio to disk and must pass timeout."""
    fake_resp = _make_fake_get_response(b"demo-audio-bytes")

    with patch.object(server, "resource_mode", "local"), \
         patch.object(server.requests, "get", return_value=fake_resp) as mock_get, \
         patch.object(server.api_client, "post") as mock_post:
        mock_post.side_effect = [
            {"file": {"file_id": "fake_file_id"}},   # /v1/files/upload (local file)
            {"demo_audio": "https://example.com/demo.wav"},  # /v1/voice_clone
        ]
        # Local file branch — write a tiny placeholder file.
        local_audio = tmp_path / "local.mp3"
        local_audio.write_bytes(b"\xff\xfb\x90\x64\x00")

        result = server.voice_clone(
            voice_id="test_voice",
            file=str(local_audio),
            text="hello",
            output_directory=str(tmp_path),
            is_url=False,
        )

    # At least one requests.get call (the demo_audio download) must use timeout.
    assert mock_get.call_count >= 1, "expected requests.get to be called"
    download_call = mock_get.call_args_list[0]
    assert download_call.kwargs.get("timeout") == DOWNLOAD_TIMEOUT, (
        f"voice_clone demo_audio download missing timeout kwarg: {download_call}"
    )


def test_generate_video_download_uses_timeout_and_stream(tmp_path):
    """generate_video() download must use timeout=DOWNLOAD_TIMEOUT and stream=True."""
    fake_resp = _make_fake_get_response(b"video-bytes")

    with patch.object(server, "resource_mode", "local"), \
         patch.object(server.requests, "get", return_value=fake_resp) as mock_get, \
         patch.object(server.api_client, "post") as mock_post, \
         patch.object(server.api_client, "get") as mock_api_get, \
         patch("minimax_mcp.server.time.sleep", return_value=None):
        # 1) /v1/video_generation submit
        # 2) (api_get below handles status polling)
        mock_post.side_effect = [
            {"task_id": "fake_task_id"},
        ]
        # Status poll returns Success+file_id; file retrieve returns download_url.
        mock_api_get.side_effect = [
            {"status": "Success", "file_id": "fake_file_id"},
            {"file": {"download_url": "https://example.com/video.mp4"}},
        ]
        result = server.generate_video(
            model="T2V-01",
            prompt="a cat",
            output_directory=str(tmp_path),
            async_mode=False,
        )

    # The final requests.get call is the video download — must be timeout + stream.
    assert mock_get.call_count == 1, f"expected one requests.get, got {mock_get.call_count}"
    download_call = mock_get.call_args_list[0]
    assert download_call.kwargs.get("timeout") == DOWNLOAD_TIMEOUT, (
        f"generate_video download missing timeout kwarg: {download_call}"
    )
    assert download_call.kwargs.get("stream") is True, (
        f"generate_video download should use stream=True for large files: {download_call}"
    )
