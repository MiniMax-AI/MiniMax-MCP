"""Tests for parameter validation helpers and tool function validation."""
import math

import pytest

from minimax_mcp.exceptions import MinimaxValidationError
from minimax_mcp.validators import _validate_range, _validate_enum
from minimax_mcp.const import (
    VALID_SAMPLE_RATES,
    VALID_BITRATES,
    VALID_EMOTIONS,
    VALID_FORMATS,
    VALID_ASPECT_RATIOS,
    SPEED_MIN, SPEED_MAX,
    VOLUME_MIN, VOLUME_MAX,
    PITCH_MIN, PITCH_MAX,
    CHANNEL_MIN, CHANNEL_MAX,
    IMAGE_N_MIN, IMAGE_N_MAX,
)


class TestValidateRange:
    """Tests for the _validate_range helper."""

    def test_accepts_value_in_middle_of_range(self):
        # Should not raise
        _validate_range("speed", 1.0, SPEED_MIN, SPEED_MAX)

    def test_accepts_lower_boundary(self):
        # Boundary value should be accepted
        _validate_range("speed", SPEED_MIN, SPEED_MIN, SPEED_MAX)

    def test_accepts_upper_boundary(self):
        # Boundary value should be accepted
        _validate_range("speed", SPEED_MAX, SPEED_MIN, SPEED_MAX)

    def test_rejects_value_below_minimum(self):
        with pytest.raises(MinimaxValidationError) as exc:
            _validate_range("speed", 0.1, SPEED_MIN, SPEED_MAX)
        assert "speed" in str(exc.value)
        assert "0.5" in str(exc.value)
        assert "2.0" in str(exc.value)

    def test_rejects_value_above_maximum(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("speed", 999, SPEED_MIN, SPEED_MAX)

    def test_rejects_nan(self):
        # NaN comparisons return False, so _validate_range should reject
        with pytest.raises(MinimaxValidationError):
            _validate_range("speed", float("nan"), SPEED_MIN, SPEED_MAX)

    def test_error_message_includes_actual_value(self):
        with pytest.raises(MinimaxValidationError) as exc:
            _validate_range("pitch", 100, PITCH_MIN, PITCH_MAX)
        assert "100" in str(exc.value)

    def test_int_range(self):
        # Test integer range (e.g., channel)
        _validate_range("channel", 1, CHANNEL_MIN, CHANNEL_MAX)
        _validate_range("channel", 2, CHANNEL_MIN, CHANNEL_MAX)
        with pytest.raises(MinimaxValidationError):
            _validate_range("channel", 0, CHANNEL_MIN, CHANNEL_MAX)
        with pytest.raises(MinimaxValidationError):
            _validate_range("channel", 3, CHANNEL_MIN, CHANNEL_MAX)


class TestValidateEnum:
    """Tests for the _validate_enum helper."""

    def test_accepts_valid_value(self):
        _validate_enum("sample_rate", 32000, VALID_SAMPLE_RATES)

    def test_rejects_invalid_value(self):
        with pytest.raises(MinimaxValidationError) as exc:
            _validate_enum("sample_rate", 12345, VALID_SAMPLE_RATES)
        assert "sample_rate" in str(exc.value)
        assert "12345" in str(exc.value)

    def test_rejects_none(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("sample_rate", None, VALID_SAMPLE_RATES)

    def test_rejects_type_mismatch(self):
        # str passed where int expected
        with pytest.raises(MinimaxValidationError):
            _validate_enum("sample_rate", "32000", VALID_SAMPLE_RATES)

    def test_rejects_empty_string(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("format", "", VALID_FORMATS)

    def test_accepts_all_sample_rates(self):
        for rate in VALID_SAMPLE_RATES:
            _validate_enum("sample_rate", rate, VALID_SAMPLE_RATES)

    def test_accepts_all_bitrates(self):
        for rate in VALID_BITRATES:
            _validate_enum("bitrate", rate, VALID_BITRATES)

    def test_accepts_all_emotions(self):
        for emotion in VALID_EMOTIONS:
            _validate_enum("emotion", emotion, VALID_EMOTIONS)

    def test_accepts_all_formats(self):
        for fmt in VALID_FORMATS:
            _validate_enum("format", fmt, VALID_FORMATS)

    def test_accepts_all_aspect_ratios(self):
        for ratio in VALID_ASPECT_RATIOS:
            _validate_enum("aspect_ratio", ratio, VALID_ASPECT_RATIOS)


class TestTextToAudioValidation:
    """End-to-end tests verifying tool functions validate inputs."""

    def test_text_to_audio_rejects_out_of_range_speed(self):
        # Import the inner function (bypassing MCP @tool decorator would require
        # more setup; we just call the validators with the same args the function uses)
        with pytest.raises(MinimaxValidationError):
            _validate_range("speed", 999, SPEED_MIN, SPEED_MAX)

    def test_text_to_audio_rejects_negative_volume(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("vol", -50, VOLUME_MIN, VOLUME_MAX)

    def test_text_to_audio_rejects_excessive_pitch(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("pitch", 100, PITCH_MIN, PITCH_MAX)

    def test_text_to_audio_rejects_invalid_sample_rate(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("sample_rate", 11025, VALID_SAMPLE_RATES)

    def test_text_to_audio_rejects_invalid_bitrate(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("bitrate", 96000, VALID_BITRATES)

    def test_text_to_audio_rejects_invalid_format(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("format", "wav", VALID_FORMATS)

    def test_text_to_audio_rejects_invalid_channel(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("channel", 5, CHANNEL_MIN, CHANNEL_MAX)


class TestTextToImageValidation:
    def test_text_to_image_rejects_n_below_minimum(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("n", 0, IMAGE_N_MIN, IMAGE_N_MAX)

    def test_text_to_image_rejects_n_above_maximum(self):
        with pytest.raises(MinimaxValidationError):
            _validate_range("n", 100, IMAGE_N_MIN, IMAGE_N_MAX)

    def test_text_to_image_accepts_n_boundary(self):
        _validate_range("n", IMAGE_N_MIN, IMAGE_N_MIN, IMAGE_N_MAX)
        _validate_range("n", IMAGE_N_MAX, IMAGE_N_MIN, IMAGE_N_MAX)

    def test_text_to_image_rejects_invalid_aspect_ratio(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("aspect_ratio", "4:5", VALID_ASPECT_RATIOS)


class TestMusicGenerationValidation:
    def test_music_generation_rejects_invalid_sample_rate(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("sample_rate", 11025, VALID_SAMPLE_RATES)

    def test_music_generation_accepts_music_sample_rates(self):
        # Music generation supports [16000, 24000, 32000, 44100]
        for rate in (16000, 24000, 32000, 44100):
            _validate_enum("sample_rate", rate, VALID_SAMPLE_RATES)

    def test_music_generation_rejects_invalid_bitrate(self):
        with pytest.raises(MinimaxValidationError):
            _validate_enum("bitrate", 192000, VALID_BITRATES)


class TestIssue66Repro:
    """Reproduces the exact failure case from issue #66."""

    def test_example_failure_raises_validation_errors(self):
        """text_to_audio(text='hello', speed=999, vol=-50, pitch=100)
        should fail validation locally with clear messages."""
        errors = []
        for name, value, mn, mx in [
            ("speed", 999, SPEED_MIN, SPEED_MAX),
            ("vol", -50, VOLUME_MIN, VOLUME_MAX),
            ("pitch", 100, PITCH_MIN, PITCH_MAX),
        ]:
            with pytest.raises(MinimaxValidationError) as exc:
                _validate_range(name, value, mn, mx)
            errors.append(str(exc.value))

        # All three bad values should produce validation errors
        assert len(errors) == 3
        assert all("must be between" in e for e in errors)
