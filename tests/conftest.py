import pytest
from pathlib import Path
import tempfile
from minimax_mcp.client import MinimaxAPIClient


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


@pytest.fixture
def api_client():
    return MinimaxAPIClient(api_key="test-key", api_host="https://api.test.com")
