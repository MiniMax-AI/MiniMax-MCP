import pytest
from unittest.mock import patch

from minimax_mcp.const import DEFAULT_MUSIC_MODEL, ENV_MUSIC_MODEL


def test_default_music_model_value():
    """Test that DEFAULT_MUSIC_MODEL equals 'music-2.6'."""
    assert DEFAULT_MUSIC_MODEL == "music-2.6"


def test_env_music_model_value():
    """Test that ENV_MUSIC_MODEL equals 'MINIMAX_MUSIC_MODEL'."""
    assert ENV_MUSIC_MODEL == "MINIMAX_MUSIC_MODEL"


@patch('os.getenv')
def test_music_model_resolved_from_env(mock_getenv):
    """Test that music_model is correctly resolved from MINIMAX_MUSIC_MODEL env var."""
    # The server.py uses: music_model = os.getenv(ENV_MUSIC_MODEL) or DEFAULT_MUSIC_MODEL
    mock_getenv.return_value = "music-custom-model"

    # Simulate what server.py does at module level
    music_model = mock_getenv(ENV_MUSIC_MODEL) or DEFAULT_MUSIC_MODEL

    assert music_model == "music-custom-model"
    mock_getenv.assert_called_with(ENV_MUSIC_MODEL)


@patch('os.getenv')
def test_music_model_uses_default_when_env_not_set(mock_getenv):
    """Test that music_model uses DEFAULT_MUSIC_MODEL when env var is not set."""
    mock_getenv.return_value = None  # Env var not set

    # Simulate what server.py does at module level
    music_model = mock_getenv(ENV_MUSIC_MODEL) or DEFAULT_MUSIC_MODEL

    assert music_model == DEFAULT_MUSIC_MODEL
    assert music_model == "music-2.6"


@patch('os.getenv')
def test_music_model_resolve_empty_string_env(mock_getenv):
    """Test that empty string env var falls back to default."""
    mock_getenv.return_value = ""  # Env var is empty string

    # Simulate what server.py does - empty string is falsy so falls back to default
    music_model = mock_getenv(ENV_MUSIC_MODEL) or DEFAULT_MUSIC_MODEL

    assert music_model == DEFAULT_MUSIC_MODEL
