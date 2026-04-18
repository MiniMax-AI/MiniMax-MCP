import pytest
import os
from minimax_mcp.exceptions import MinimaxRequestError


@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setenv("MINIMAX_API_HOST", "https://api.test.com")


def test_validate_speed_too_high():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="speed"):
        _validate_audio_params(speed=3.0)


def test_validate_speed_too_low():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="speed"):
        _validate_audio_params(speed=0.1)


def test_validate_vol_out_of_range():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="vol"):
        _validate_audio_params(vol=-1)


def test_validate_pitch_out_of_range():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="pitch"):
        _validate_audio_params(pitch=20)


def test_validate_invalid_sample_rate():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="sample_rate"):
        _validate_audio_params(sample_rate=99999)


def test_validate_invalid_bitrate():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="bitrate"):
        _validate_audio_params(bitrate=1)


def test_validate_invalid_audio_format():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="audio_format"):
        _validate_audio_params(audio_format="ogg")


def test_validate_n_out_of_range():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="n"):
        _validate_audio_params(n=0)


def test_validate_invalid_aspect_ratio():
    from minimax_mcp.server import _validate_audio_params

    with pytest.raises(MinimaxRequestError, match="aspect_ratio"):
        _validate_audio_params(aspect_ratio="5:3")


def test_validate_valid_params_passes():
    from minimax_mcp.server import _validate_audio_params

    # Should not raise
    _validate_audio_params(
        speed=1.0, vol=5, pitch=0, sample_rate=32000,
        bitrate=128000, channel=1, audio_format="mp3", n=3,
        aspect_ratio="16:9"
    )
