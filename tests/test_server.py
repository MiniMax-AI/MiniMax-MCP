"""Tests for minimax_mcp/server.py.

These tests mock the `api_client` module-level instance, the `requests.get`
HTTP calls, and the `play` audio helper so we can exercise the public MCP
tool surface end-to-end without hitting the network or audio hardware.

The tests do NOT change any production code; they only verify behavior.
"""

import base64
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open

import pytest

# Import the server module. conftest.py ensures MINIMAX_API_KEY and
# MINIMAX_API_HOST are set in os.environ before pytest collects this file.
from minimax_mcp import server
from minimax_mcp.exceptions import MinimaxAPIError, MinimaxRequestError
from minimax_mcp.const import RESOURCE_MODE_URL, RESOURCE_MODE_LOCAL


# ----------------------------------------------------------------------------
# Helpers / fixtures
# ----------------------------------------------------------------------------


@pytest.fixture
def local_mode(monkeypatch):
    """Switch the server into local-file resource mode for the duration of a test."""
    monkeypatch.setattr(server, "resource_mode", RESOURCE_MODE_LOCAL)


@pytest.fixture
def url_mode(monkeypatch):
    """Force URL resource mode (the server's default, but explicit)."""
    monkeypatch.setattr(server, "resource_mode", RESOURCE_MODE_URL)


@pytest.fixture
def base_dir(temp_dir, monkeypatch):
    """Point the server's base_path at a writable temp dir."""
    monkeypatch.setattr(server, "base_path", str(temp_dir))
    return temp_dir


@pytest.fixture
def api_client_mock(monkeypatch):
    """Patch the module-level api_client with a MagicMock."""
    mock = MagicMock()
    monkeypatch.setattr(server, "api_client", mock)
    return mock


@pytest.fixture
def hex_audio():
    """Sample audio bytes encoded as a hex string (matching server.py's expectation)."""
    return b"\x00\x01\x02\x03\x04 audio bytes \xff".hex()


def _make_response(content: bytes = b"binary-data"):
    """Build a fake `requests.get(...).content` response object."""
    resp = MagicMock()
    resp.content = content
    resp.raise_for_status = MagicMock()
    return resp


# ----------------------------------------------------------------------------
# text_to_audio
# ----------------------------------------------------------------------------


class TestTextToAudio:
    def test_empty_text_raises_request_error(self):
        """The `text` parameter is required and not in a try/except — must raise."""
        with pytest.raises(MinimaxRequestError, match="Text is required"):
            server.text_to_audio(text="")

    def test_none_text_raises_request_error(self):
        with pytest.raises(MinimaxRequestError, match="Text is required"):
            server.text_to_audio(text=None)

    def test_url_mode_returns_audio_url(self, url_mode, api_client_mock, hex_audio):
        api_client_mock.post.return_value = {"data": {"audio": "https://cdn.example/audio.mp3"}}

        result = server.text_to_audio(text="hello world")

        api_client_mock.post.assert_called_once()
        assert "Audio URL: https://cdn.example/audio.mp3" in result.text
        assert result.type == "text"

    def test_local_mode_saves_audio_file(self, local_mode, base_dir, api_client_mock, hex_audio):
        api_client_mock.post.return_value = {"data": {"audio": hex_audio}}

        result = server.text_to_audio(text="hello world", output_directory=str(base_dir))

        # File should have been written somewhere under base_dir
        api_client_mock.post.assert_called_once()
        assert "Success. File saved as:" in result.text
        assert "Voice used:" in result.text

        # Confirm at least one t2a_*.mp3 file exists in the output directory
        files = list(base_dir.glob("t2a_*.mp3"))
        assert len(files) == 1
        # And the contents match the hex-decoded bytes
        assert files[0].read_bytes() == bytes.fromhex(hex_audio)

    def test_api_error_returns_textcontent(self, url_mode, api_client_mock):
        api_client_mock.post.side_effect = MinimaxAPIError("auth failed")

        result = server.text_to_audio(text="hello")

        assert "Failed to generate audio" in result.text
        assert "auth failed" in result.text
        assert result.type == "text"

    def test_empty_audio_data_caught_and_returned_as_text(self, url_mode, api_client_mock):
        """An empty `audio` field raises MinimaxRequestError inside a try/except
        MinimaxAPIError — so the function should catch it and return a
        TextContent rather than propagating."""
        api_client_mock.post.return_value = {"data": {"audio": ""}}

        result = server.text_to_audio(text="hello")

        assert result.type == "text"
        # Implementation prints the raw exception text, which is informative
        # for an MCP tool caller. We just verify the failure was surfaced.
        assert "Failed" in result.text


# ----------------------------------------------------------------------------
# list_voices
# ----------------------------------------------------------------------------


class TestListVoices:
    def test_returns_system_and_cloning_voices(self, api_client_mock):
        api_client_mock.post.return_value = {
            "system_voice": [
                {"voice_name": "Alloy", "voice_id": "alloy"},
                {"voice_name": "Echo", "voice_id": "echo"},
            ],
            "voice_cloning": [
                {"voice_name": "MyClone", "voice_id": "my_clone"},
            ],
        }

        result = server.list_voices()

        assert "Alloy" in result.text
        assert "alloy" in result.text
        assert "MyClone" in result.text
        assert result.type == "text"

    def test_handles_none_lists(self, api_client_mock):
        """The implementation defaults None entries to []. Make sure we don't crash."""
        api_client_mock.post.return_value = {"system_voice": None, "voice_cloning": None}

        result = server.list_voices()

        assert "System Voices: []" in result.text
        assert "Voice Cloning Voices: []" in result.text

    def test_handles_missing_keys(self, api_client_mock):
        """The response may omit both keys entirely; should not crash."""
        api_client_mock.post.return_value = {}

        result = server.list_voices()

        assert result.type == "text"
        assert "System Voices" in result.text

    def test_api_error_returns_textcontent(self, api_client_mock):
        api_client_mock.post.side_effect = MinimaxAPIError("network down")

        result = server.list_voices()

        assert "Failed to list voices" in result.text
        assert "network down" in result.text


# ----------------------------------------------------------------------------
# voice_clone
# ----------------------------------------------------------------------------


class TestVoiceClone:
    def test_local_file_upload_and_clone(self, base_dir, api_client_mock, monkeypatch):
        """Upload a local audio file, then clone, no demo audio in response."""
        audio = base_dir / "source.mp3"
        audio.write_bytes(b"source-audio-bytes")

        # /v1/files/upload returns file_id; /v1/voice_clone returns no demo_audio
        api_client_mock.post.side_effect = [
            {"file": {"file_id": "fid-123"}},
            {"demo_audio": ""},  # no demo audio
        ]

        result = server.voice_clone(
            voice_id="my-voice",
            file=str(audio),
            text="clone this",
            output_directory=str(base_dir),
        )

        assert api_client_mock.post.call_count == 2
        assert "Voice cloned successfully" in result.text
        assert "my-voice" in result.text

    def test_local_file_not_found_returns_error(self, base_dir, api_client_mock):
        result = server.voice_clone(
            voice_id="my-voice",
            file=str(base_dir / "missing.mp3"),
            text="clone this",
            output_directory=str(base_dir),
        )

        # Implementation raises MinimaxRequestError, caught, returns TextContent
        assert "Failed to clone voice" in result.text
        assert "does not exist" in result.text
        api_client_mock.post.assert_not_called()

    def test_url_path_uploads_from_url(self, api_client_mock, base_dir):
        api_client_mock.post.side_effect = [
            {"file": {"file_id": "fid-url"}},
            {"demo_audio": ""},  # no demo audio
        ]

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"downloaded-bytes")

            result = server.voice_clone(
                voice_id="url-voice",
                file="https://example.com/sample.mp3",
                text="clone this",
                is_url=True,
                output_directory=str(base_dir),
            )

        assert api_client_mock.post.call_count == 2
        assert "Voice cloned successfully" in result.text

    def test_demo_audio_local_mode_downloads_and_saves(
        self, base_dir, local_mode, api_client_mock, monkeypatch
    ):
        audio = base_dir / "source.mp3"
        audio.write_bytes(b"source-audio-bytes")

        # Force local resource mode (the fixture handles it, just being explicit)
        monkeypatch.setattr(server, "resource_mode", RESOURCE_MODE_LOCAL)
        api_client_mock.post.side_effect = [
            {"file": {"file_id": "fid-456"}},
            {"demo_audio": "https://cdn.example/demo.wav"},
        ]

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"demo-audio-bytes")

            result = server.voice_clone(
                voice_id="voice-with-demo",
                file=str(audio),
                text="clone this",
                output_directory=str(base_dir),
            )

        assert "demo audio saved" in result.text
        # Confirm the demo audio file landed on disk
        wav_files = list(base_dir.glob("voice_clone_*.wav"))
        assert len(wav_files) == 1
        assert wav_files[0].read_bytes() == b"demo-audio-bytes"

    def test_demo_audio_url_mode_returns_url(self, base_dir, url_mode, api_client_mock):
        audio = base_dir / "source.mp3"
        audio.write_bytes(b"source-audio-bytes")
        api_client_mock.post.side_effect = [
            {"file": {"file_id": "fid-789"}},
            {"demo_audio": "https://cdn.example/demo.wav"},
        ]

        result = server.voice_clone(
            voice_id="voice-with-demo",
            file=str(audio),
            text="clone this",
            output_directory=str(base_dir),
        )

        assert "Demo audio URL: https://cdn.example/demo.wav" in result.text
        # Should NOT have downloaded the demo audio
        wav_files = list(base_dir.glob("voice_clone_*.wav"))
        assert wav_files == []

    def test_missing_file_id_raises_request_error(self, base_dir, api_client_mock):
        audio = base_dir / "source.mp3"
        audio.write_bytes(b"x")
        api_client_mock.post.return_value = {"file": {}}  # no file_id

        result = server.voice_clone(
            voice_id="voice",
            file=str(audio),
            text="clone this",
        )

        # Caught, returned as TextContent
        assert "Failed to clone voice" in result.text
        assert "file_id" in result.text


# ----------------------------------------------------------------------------
# play_audio
# ----------------------------------------------------------------------------


class TestPlayAudio:
    def test_local_file_calls_play(self, base_dir, api_client_mock, sample_audio_file, monkeypatch):
        # sample_audio_file lives in its own tempdir; use that
        monkeypatch.setattr(server, "base_path", str(sample_audio_file.parent))
        sample_audio_file.write_bytes(b"mp3bytes")

        with patch("minimax_mcp.server.play") as mock_play:
            result = server.play_audio(input_file_path=str(sample_audio_file))

        mock_play.assert_called_once_with(b"mp3bytes")
        assert "Successfully played audio file" in result.text
        assert str(sample_audio_file) in result.text

    def test_url_path_downloads_then_plays(self, api_client_mock):
        with patch("minimax_mcp.server.play") as mock_play, \
             patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"streamed-bytes")

            result = server.play_audio(
                input_file_path="https://example.com/song.mp3",
                is_url=True,
            )

        mock_get.assert_called_once_with("https://example.com/song.mp3")
        mock_play.assert_called_once_with(b"streamed-bytes")
        assert "Successfully played audio file" in result.text


# ----------------------------------------------------------------------------
# generate_video
# ----------------------------------------------------------------------------


class TestGenerateVideo:
    def test_empty_prompt_returns_error_text(self):
        """`Prompt is required` is raised inside the try/except block, so it
        is caught and returned as TextContent (not propagated)."""
        result = server.generate_video(prompt="")

        assert result.type == "text"
        assert "Prompt is required" in result.text

    def test_first_frame_must_be_string(self, base_dir, api_client_mock):
        """Non-string first_frame_image raises inside the try/except — caught."""
        result = server.generate_video(prompt="a cat", first_frame_image=12345)

        assert result.type == "text"
        assert "must be a string" in result.text

    def test_first_frame_local_file_must_exist(self, base_dir, api_client_mock):
        result = server.generate_video(
            prompt="a cat",
            first_frame_image=str(base_dir / "missing.jpg"),
        )

        assert result.type == "text"
        assert "does not exist" in result.text

    def test_first_frame_local_file_is_base64_encoded(
        self, base_dir, local_mode, api_client_mock, monkeypatch
    ):
        img = base_dir / "frame.jpg"
        img.write_bytes(b"jpeg-bytes")
        api_client_mock.post.return_value = {"task_id": "task-1"}
        # First get: status poll (Success). Second get: file retrieve.
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-1"},
            {"file": {"download_url": "https://cdn.example/video.mp4"}},
        ]
        monkeypatch.setattr(server, "time", MagicMock())  # make sleep a no-op

        # Mock download to avoid network
        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"video-bytes")
            result = server.generate_video(
                prompt="a cat",
                first_frame_image=str(img),
                output_directory=str(base_dir),
            )

        # Verify the first_frame_image in payload was a data URL
        call = api_client_mock.post.call_args_list[0]
        payload = call.kwargs.get("json") or call.args[1]
        assert payload["first_frame_image"].startswith("data:image/jpeg;base64,")
        # The base64 part should decode to the original bytes
        b64 = payload["first_frame_image"].split(",", 1)[1]
        assert base64.b64decode(b64) == b"jpeg-bytes"
        assert "Video saved as:" in result.text

    def test_first_frame_http_url_passed_through(
        self, base_dir, local_mode, api_client_mock, monkeypatch
    ):
        api_client_mock.post.return_value = {"task_id": "task-1"}
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-1"},
            {"file": {"download_url": "https://cdn.example/video.mp4"}},
        ]
        monkeypatch.setattr(server, "time", MagicMock())

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"video-bytes")
            server.generate_video(
                prompt="a cat",
                first_frame_image="https://cdn.example/frame.jpg",
                output_directory=str(base_dir),
            )

        call = api_client_mock.post.call_args_list[0]
        payload = call.kwargs.get("json") or call.args[1]
        assert payload["first_frame_image"] == "https://cdn.example/frame.jpg"

    def test_async_mode_returns_task_id_without_polling(
        self, base_dir, api_client_mock
    ):
        api_client_mock.post.return_value = {"task_id": "task-async"}

        result = server.generate_video(
            prompt="a cat", async_mode=True, output_directory=str(base_dir)
        )

        # Should NOT have called api_client.get
        api_client_mock.get.assert_not_called()
        assert "task-async" in result.text
        assert "query_video_generation" in result.text

    def test_sync_mode_polls_until_success(
        self, base_dir, local_mode, api_client_mock, monkeypatch
    ):
        api_client_mock.post.return_value = {"task_id": "task-sync"}
        # Three get calls: poll (Processing), poll (Success), file retrieve
        api_client_mock.get.side_effect = [
            {"status": "Processing"},
            {"status": "Success", "file_id": "fid-final"},
            {"file": {"download_url": "https://cdn.example/video.mp4"}},
        ]
        monkeypatch.setattr(server, "time", MagicMock())

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"video-bytes")
            result = server.generate_video(
                prompt="a cat", output_directory=str(base_dir)
            )

        assert "Video saved as:" in result.text
        mp4_files = list(base_dir.glob("video_*.mp4"))
        assert len(mp4_files) == 1

    def test_url_mode_returns_video_url(self, base_dir, url_mode, api_client_mock, monkeypatch):
        api_client_mock.post.return_value = {"task_id": "task-url"}
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-url"},
            {"file": {"download_url": "https://cdn.example/video.mp4"}},
        ]
        monkeypatch.setattr(server, "time", MagicMock())

        result = server.generate_video(
            prompt="a cat", output_directory=str(base_dir)
        )

        assert "Video URL: https://cdn.example/video.mp4" in result.text
        # No file should have been written
        mp4_files = list(base_dir.glob("video_*.mp4"))
        assert mp4_files == []

    def test_poll_failure_returns_error_text(self, base_dir, api_client_mock, monkeypatch):
        api_client_mock.post.return_value = {"task_id": "task-fail"}
        api_client_mock.get.return_value = {"status": "Fail"}
        monkeypatch.setattr(server, "time", MagicMock())

        result = server.generate_video(
            prompt="a cat", output_directory=str(base_dir)
        )

        assert "Failed to generate video" in result.text
        assert "task-fail" in result.text

    def test_missing_task_id_in_submit(self, base_dir, api_client_mock):
        api_client_mock.post.return_value = {}  # no task_id

        result = server.generate_video(
            prompt="a cat", output_directory=str(base_dir)
        )

        assert "Failed to generate video" in result.text

    def test_missing_download_url(self, base_dir, api_client_mock, monkeypatch):
        api_client_mock.post.return_value = {"task_id": "task-no-url"}
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-nourl"},
            {"file": {}},  # no download_url
        ]
        monkeypatch.setattr(server, "time", MagicMock())

        result = server.generate_video(
            prompt="a cat", output_directory=str(base_dir)
        )

        assert "Failed to generate video" in result.text
        assert "download URL" in result.text


# ----------------------------------------------------------------------------
# query_video_generation
# ----------------------------------------------------------------------------


class TestQueryVideoGeneration:
    def test_status_processing_returns_status_text(self, api_client_mock):
        api_client_mock.get.return_value = {"status": "Processing"}

        result = server.query_video_generation(task_id="t-1")

        assert "still processing" in result.text
        assert "t-1" in result.text

    def test_status_fail_returns_failure_text(self, api_client_mock):
        api_client_mock.get.return_value = {"status": "Fail"}

        result = server.query_video_generation(task_id="t-2")

        assert "FAILED" in result.text
        assert "t-2" in result.text

    def test_status_success_url_mode(self, base_dir, url_mode, api_client_mock):
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-1"},
            {"file": {"download_url": "https://cdn.example/v.mp4"}},
        ]

        result = server.query_video_generation(task_id="t-3", output_directory=str(base_dir))

        assert "Video URL: https://cdn.example/v.mp4" in result.text

    def test_status_success_local_mode_saves_file(
        self, base_dir, local_mode, api_client_mock
    ):
        api_client_mock.get.side_effect = [
            {"status": "Success", "file_id": "fid-2"},
            {"file": {"download_url": "https://cdn.example/v.mp4"}},
        ]

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"video-bytes")
            result = server.query_video_generation(
                task_id="t-4", output_directory=str(base_dir)
            )

        assert "Video saved as:" in result.text
        assert any(base_dir.glob("video_*.mp4"))


# ----------------------------------------------------------------------------
# text_to_image
# ----------------------------------------------------------------------------


class TestTextToImage:
    def test_empty_prompt_returns_error_text(self):
        """Prompt validation lives inside the try/except — caught as TextContent."""
        result = server.text_to_image(prompt="")

        assert result.type == "text"
        assert "Prompt is required" in result.text

    def test_url_mode_returns_image_urls(self, url_mode, api_client_mock):
        api_client_mock.post.return_value = {
            "data": {"image_urls": ["https://cdn.example/a.jpg", "https://cdn.example/b.jpg"]}
        }

        result = server.text_to_image(prompt="a cat")

        assert "https://cdn.example/a.jpg" in result.text
        assert "https://cdn.example/b.jpg" in result.text

    def test_local_mode_downloads_and_saves_images(
        self, base_dir, local_mode, api_client_mock
    ):
        api_client_mock.post.return_value = {
            "data": {"image_urls": ["https://cdn.example/a.jpg", "https://cdn.example/b.jpg"]}
        }

        with patch("minimax_mcp.server.requests.get") as mock_get:
            mock_get.return_value = _make_response(b"image-bytes")
            result = server.text_to_image(
                prompt="a cat", output_directory=str(base_dir)
            )

        assert "Images saved as:" in result.text
        jpg_files = sorted(base_dir.glob("image_*.jpg"))
        assert len(jpg_files) == 2
        for f in jpg_files:
            assert f.read_bytes() == b"image-bytes"

    def test_no_images_in_response_returns_error_text(self, url_mode, api_client_mock):
        api_client_mock.post.return_value = {"data": {"image_urls": []}}

        result = server.text_to_image(prompt="a cat")

        assert result.type == "text"
        assert "No images generated" in result.text

    def test_api_error_returns_textcontent(self, url_mode, api_client_mock):
        api_client_mock.post.side_effect = MinimaxAPIError("rate limit")

        result = server.text_to_image(prompt="a cat")

        assert "Failed to generate images" in result.text
        assert "rate limit" in result.text


# ----------------------------------------------------------------------------
# music_generation
# ----------------------------------------------------------------------------


class TestMusicGeneration:
    def test_empty_prompt_returns_error_text(self):
        result = server.music_generation(prompt="", lyrics="la la la")

        assert result.type == "text"
        assert "Prompt is required" in result.text

    def test_empty_lyrics_returns_error_text(self):
        result = server.music_generation(prompt="pop", lyrics="")

        assert result.type == "text"
        assert "Lyrics is required" in result.text

    def test_url_mode_returns_music_url(self, url_mode, api_client_mock):
        api_client_mock.post.return_value = {
            "data": {"audio": "https://cdn.example/song.mp3"}
        }

        result = server.music_generation(prompt="pop", lyrics="la la la")

        assert "Music url: https://cdn.example/song.mp3" in result.text

    def test_local_mode_saves_music_file(
        self, base_dir, local_mode, api_client_mock, hex_audio
    ):
        api_client_mock.post.return_value = {"data": {"audio": hex_audio}}

        result = server.music_generation(
            prompt="pop", lyrics="la la la", output_directory=str(base_dir)
        )

        assert "Music saved as:" in result.text
        files = list(base_dir.glob("music_*.mp3"))
        assert len(files) == 1
        assert files[0].read_bytes() == bytes.fromhex(hex_audio)

    def test_api_error_returns_textcontent(self, url_mode, api_client_mock):
        api_client_mock.post.side_effect = MinimaxAPIError("quota")

        result = server.music_generation(prompt="pop", lyrics="la la la")

        assert "Failed to generate music" in result.text
        assert "quota" in result.text


# ----------------------------------------------------------------------------
# voice_design
# ----------------------------------------------------------------------------


class TestVoiceDesign:
    def test_empty_prompt_returns_error_text(self):
        result = server.voice_design(prompt="", preview_text="hello world")

        assert result.type == "text"
        assert "prompt is required" in result.text

    def test_empty_preview_text_returns_error_text(self):
        result = server.voice_design(prompt="a voice", preview_text="")

        assert result.type == "text"
        assert "preview_text is required" in result.text

    def test_url_mode_returns_voice_info(self, url_mode, api_client_mock):
        api_client_mock.post.return_value = {
            "voice_id": "new-voice-id",
            "trial_audio": "https://cdn.example/trial.mp3",
        }

        result = server.voice_design(prompt="a calm voice", preview_text="hello")

        assert "new-voice-id" in result.text
        assert "https://cdn.example/trial.mp3" in result.text

    def test_local_mode_saves_audio(
        self, base_dir, local_mode, api_client_mock, hex_audio
    ):
        api_client_mock.post.return_value = {
            "voice_id": "v-1",
            "trial_audio": hex_audio,
        }

        result = server.voice_design(
            prompt="a calm voice", preview_text="hello", output_directory=str(base_dir)
        )

        assert "File saved as:" in result.text
        assert "v-1" in result.text
        files = list(base_dir.glob("voice_design_*.mp3"))
        assert len(files) == 1
        assert files[0].read_bytes() == bytes.fromhex(hex_audio)

    def test_no_voice_id_returns_error(self, url_mode, api_client_mock):
        api_client_mock.post.return_value = {"voice_id": ""}

        result = server.voice_design(prompt="a voice", preview_text="hello")

        # Caught MinimaxRequestError, returned as TextContent
        assert "Failed to design voice" in result.text
        assert "No voice generated" in result.text

    def test_voice_id_included_in_payload_when_provided(
        self, url_mode, api_client_mock
    ):
        api_client_mock.post.return_value = {
            "voice_id": "v-1",
            "trial_audio": "https://x.example/t.mp3",
        }

        server.voice_design(
            prompt="a voice", preview_text="hello", voice_id="custom-vid"
        )

        call = api_client_mock.post.call_args
        payload = call.kwargs.get("json") or call.args[1]
        assert payload["voice_id"] == "custom-vid"

    def test_voice_id_omitted_from_payload_when_none(
        self, url_mode, api_client_mock
    ):
        api_client_mock.post.return_value = {
            "voice_id": "v-1",
            "trial_audio": "https://x.example/t.mp3",
        }

        server.voice_design(prompt="a voice", preview_text="hello", voice_id=None)

        call = api_client_mock.post.call_args
        payload = call.kwargs.get("json") or call.args[1]
        assert "voice_id" not in payload


# ----------------------------------------------------------------------------
# main() entry point
# ----------------------------------------------------------------------------


class TestServerMain:
    def test_main_calls_mcp_run(self, capsys):
        with patch("minimax_mcp.server.mcp") as mock_mcp:
            server.main()

        mock_mcp.run.assert_called_once()
        captured = capsys.readouterr()
        assert "Starting Minimax MCP server" in captured.out
