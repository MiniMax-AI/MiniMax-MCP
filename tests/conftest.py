import os

# Set environment variables required by minimax_mcp.server at import time.
# server.py raises ValueError at module load if MINIMAX_API_KEY or
# MINIMAX_API_HOST are not set, so we must set them before pytest
# collects/imports any test module that transitively imports server.
os.environ.setdefault("MINIMAX_API_KEY", "test-api-key")
os.environ.setdefault("MINIMAX_API_HOST", "https://api.test.example")

import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_audio_file(temp_dir):
    audio_file = temp_dir / "test.mp3"
    audio_file.touch()
    return audio_file


@pytest.fixture
def sample_video_file(temp_dir):
    video_file = temp_dir / "test.mp4"
    video_file.touch()
    return video_file
