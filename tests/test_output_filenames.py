from datetime import datetime

import pytest

from minimax_mcp import utils


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch):
    class FixedDatetime:
        @staticmethod
        def now():
            return datetime(2026, 10, 3, 8, 0, 0)

    monkeypatch.setattr(utils, "datetime", FixedDatetime)


@pytest.mark.parametrize(
    "text, component",
    [
        ("2026/10/03", "2026_10_03"),
        (r"2026\10\03", "2026_10_03"),
        ("../notes", ".._notes"),
        (r"..\notes", ".._notes"),
    ],
)
def test_text_separators_do_not_create_subdirectories(tmp_path, text, component):
    output = utils.build_output_file("t2a", text, tmp_path, "mp3")
    assert output == tmp_path / f"t2a_{component}_20261003_080000.mp3"
    output.write_bytes(b"audio write probe")
    assert output.read_bytes() == b"audio write probe"
    assert list(tmp_path.iterdir()) == [output]


@pytest.mark.parametrize("character", '<>:"|?*\x00\x01\t\n\x1f')
def test_invalid_filename_characters_allow_actual_writes(tmp_path, character):
    output = utils.build_output_file("t2a", f"a{character}b", tmp_path, "mp3")
    output.write_bytes(b"probe")
    assert output == tmp_path / "t2a_a_b_20261003_080000.mp3"
    assert output.read_bytes() == b"probe"


@pytest.mark.parametrize(
    "text, full_id, component",
    [
        ("hello world", False, "hello_worl"),
        ("你好，世界🙂", False, "你好，世界🙂"),
        ("2026-10-03", False, "2026-10-03"),
        ("", False, ""),
        ("CON", False, "CON"),
        ("...", False, "..."),
        ("long-task-id-123456789", True, "long-task-id-123456789"),
    ],
)
def test_existing_valid_names_stay_the_same(tmp_path, text, full_id, component):
    output = utils.build_output_file("video", text, tmp_path, "mp4", full_id)
    assert output == tmp_path / f"video_{component}_20261003_080000.mp4"
    output.write_bytes(b"probe")
    assert output.read_bytes() == b"probe"


def test_full_id_keeps_and_sanitizes_characters_after_the_tenth(tmp_path):
    output = utils.build_output_file(
        "video", "1234567890/part:2", tmp_path, "mp4", True
    )
    assert output == tmp_path / "video_1234567890_part_2_20261003_080000.mp4"
    output.write_bytes(b"probe")
    assert output.read_bytes() == b"probe"


def test_default_id_still_uses_only_the_first_ten_characters(tmp_path):
    output = utils.build_output_file("t2a", "1234567890/ignored", tmp_path, "mp3")
    assert output == tmp_path / "t2a_1234567890_20261003_080000.mp3"


@pytest.mark.parametrize(
    "tool", ["t2a", "voice_clone", "video", "image", "voice_design"]
)
def test_shared_tool_prefixes_keep_the_selected_output_directory(tmp_path, tool):
    directory = tmp_path / "保存 文件"
    directory.mkdir()
    output = utils.build_output_file(tool, "2026/10/03", directory, "bin")
    assert output.parent == directory
    assert output.name.startswith(tool + "_2026_10_03_")
    output.write_bytes(b"probe")
    assert output.read_bytes() == b"probe"
