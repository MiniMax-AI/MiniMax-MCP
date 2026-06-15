import pytest
from pathlib import Path
import tempfile
from minimax_mcp.utils import (
    MinimaxMcpError,
    is_file_writeable,
    build_output_file,
    build_output_path,
    find_similar_filenames,
    try_find_similar_files,
    process_input_file,
)

def test_is_file_writeable():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        assert is_file_writeable(temp_path) is True
        assert is_file_writeable(temp_path / "nonexistent.txt") is True
        # Multi-level new path: the immediate parent does not exist yet, but
        # mkdir(parents=True) can create it, so it must be reported writeable
        # (regression: only the immediate parent used to be checked).
        assert is_file_writeable(temp_path / "a" / "b" / "c") is True


def test_build_output_path_creates_nested_dir():
    # A nested, not-yet-existing output directory must be created rather than
    # rejected as "not writeable".
    with tempfile.TemporaryDirectory() as temp_dir:
        result = build_output_path("a/b/c", temp_dir)
        assert result == Path(temp_dir) / "a" / "b" / "c"
        assert result.exists()
        assert result.is_dir()


def test_make_output_file():
    tool = "test"
    text = "hello world"
    output_path = Path("/tmp")
    result = build_output_file(tool, text, output_path, "mp3")
    assert result.name.startswith("test_hello")
    assert result.suffix == ".mp3"


def test_make_output_path():
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        result = build_output_path(temp_dir)
        assert result == Path(temp_dir)
        assert result.exists()
        assert result.is_dir()

    # Test with None output_directory (should use base_path)
    base_path = "/tmp/test_base"
    result = build_output_path(None, base_path, is_test=True)
    assert result == Path(base_path)
    
    # Test with relative output_directory
    base_path = "/tmp/test_base"
    result = build_output_path("subdir", base_path, is_test=True)
    assert result == Path(base_path) / "subdir"
    
    # Test with absolute output_directory (should ignore base_path)
    abs_path = "/absolute/path"
    result = build_output_path(abs_path, "/some/base/path", is_test=True)
    assert result == Path(abs_path)

    abs_path = "~/absolute/path"
    result = build_output_path(abs_path, "/some/base/path", is_test=True)
    assert result == Path(Path.home() / "absolute/path")
    
    # Test with None base_path (should use desktop)
    result = build_output_path(None, None, is_test=True)
    assert result == Path.home() / "Desktop"
    


def test_find_similar_filenames():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.txt"
        similar_file = temp_path / "test_file_2.txt"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = find_similar_filenames(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r[0]) for r in results)


def test_try_find_similar_files():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test_file.mp3"
        similar_file = temp_path / "test_file_2.mp3"
        different_file = temp_path / "different.txt"

        test_file.touch()
        similar_file.touch()
        different_file.touch()

        results = try_find_similar_files(str(test_file), temp_path)
        assert len(results) > 0
        assert any(str(similar_file) in str(r) for r in results)


def test_process_input_file():
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        test_file = temp_path / "test.mp3"

        with open(test_file, "wb") as f:
            f.write(b"\xff\xfb\x90\x64\x00")

        result = process_input_file(str(test_file))
        assert result == test_file

        with pytest.raises(MinimaxMcpError):
            process_input_file(str(temp_path / "nonexistent.mp3"))
