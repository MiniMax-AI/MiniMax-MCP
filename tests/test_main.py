"""Tests for minimax_mcp/__main__.py.

The __main__ module is a small CLI for generating the Claude MCP config
file. Tests focus on:
- Module imports cleanly (smoke)
- get_claude_config_path handles all platforms and the missing-dir case
- generate_config builds the right config from env or arg
- generate_config exits when no API key is available
- The argparse-driven `if __name__ == "__main__"` block handles --print and --config-path
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# Import lazily inside tests where possible to avoid module-load side effects
# (argparse runs at import time in `if __name__ == "__main__"` so we must
# avoid triggering it during test collection). Importing the module directly
# only loads the function definitions.
from minimax_mcp import __main__ as mcp_main


# ----------------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------------


def test_module_imports():
    """The module should be importable as minimax_mcp.__main__ without errors."""
    assert mcp_main is not None


def test_module_exposes_expected_callables():
    """Verify the entry-point symbols are exposed for external callers."""
    assert callable(mcp_main.get_claude_config_path)
    assert callable(mcp_main.get_python_path)
    assert callable(mcp_main.generate_config)


# ----------------------------------------------------------------------------
# get_claude_config_path
# ----------------------------------------------------------------------------


def test_get_claude_config_path_darwin_existing(monkeypatch, tmp_path):
    fake = tmp_path / "Library" / "Application Support" / "Claude"
    fake.mkdir(parents=True)

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = mcp_main.get_claude_config_path()
    assert result == fake


def test_get_claude_config_path_darwin_missing(monkeypatch, tmp_path):
    """A macOS home without the Claude dir should return None (not raise)."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    assert mcp_main.get_claude_config_path() is None


def test_get_claude_config_path_linux_uses_xdg(monkeypatch, tmp_path):
    fake = tmp_path / "config" / "Claude"
    fake.mkdir(parents=True)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    result = mcp_main.get_claude_config_path()
    assert result == fake


def test_get_claude_config_path_linux_falls_back_to_home(monkeypatch, tmp_path):
    """On Linux without XDG_CONFIG_HOME, fall back to ~/.config/Claude."""
    fake = tmp_path / ".config" / "Claude"
    fake.mkdir(parents=True)

    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = mcp_main.get_claude_config_path()
    assert result == fake


def test_get_claude_config_path_windows(monkeypatch, tmp_path):
    fake = tmp_path / "AppData" / "Roaming" / "Claude"
    fake.mkdir(parents=True)

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    result = mcp_main.get_claude_config_path()
    assert result == fake


def test_get_claude_config_path_unsupported_platform(monkeypatch):
    monkeypatch.setattr(sys, "platform", "plan9")
    # No need to mock home, the function should bail early
    assert mcp_main.get_claude_config_path() is None


# ----------------------------------------------------------------------------
# get_python_path
# ----------------------------------------------------------------------------


def test_get_python_path_returns_sys_executable():
    assert mcp_main.get_python_path() == sys.executable


# ----------------------------------------------------------------------------
# generate_config
# ----------------------------------------------------------------------------


def test_generate_config_uses_explicit_api_key(monkeypatch):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    config = mcp_main.generate_config(api_key="explicit-key")

    servers = config["mcpServers"]["Minimax"]
    assert servers["env"]["MINIMAX_API_KEY"] == "explicit-key"
    assert servers["command"] == "uvx"
    assert servers["args"] == ["minimax-mcp"]


def test_generate_config_falls_back_to_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    config = mcp_main.generate_config()

    servers = config["mcpServers"]["Minimax"]
    assert servers["env"]["MINIMAX_API_KEY"] == "env-key"


def test_generate_config_explicit_arg_overrides_env(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "env-key")

    config = mcp_main.generate_config(api_key="arg-key")

    servers = config["mcpServers"]["Minimax"]
    assert servers["env"]["MINIMAX_API_KEY"] == "arg-key"


def test_generate_config_no_key_exits(monkeypatch, capsys):
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)

    with pytest.raises(SystemExit) as excinfo:
        mcp_main.generate_config()

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "API key is required" in captured.out


def test_generate_config_has_expected_env_defaults(monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "k")

    config = mcp_main.generate_config()

    env = config["mcpServers"]["Minimax"]["env"]
    assert env["MINIMAX_MCP_BASE_PATH"] == ""
    assert env["MINIMAX_API_HOST"] == "https://api.minimax.chat"


# ----------------------------------------------------------------------------
# __main__ block (argparse paths)
# ----------------------------------------------------------------------------


def test_main_block_with_print_flag(monkeypatch, capsys):
    """`python -m minimax_mcp --print --api-key k` should print config JSON."""
    monkeypatch.setattr(sys, "argv", ["__main__.py", "--print", "--api-key", "k"])

    # Re-run the module body in __main__ namespace as a fresh import
    runpy = pytest.importorskip("runpy")
    runpy.run_module(
        "minimax_mcp.__main__", run_name="__main__", alter_sys=True
    )

    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert parsed["mcpServers"]["Minimax"]["env"]["MINIMAX_API_KEY"] == "k"


def test_main_block_writes_to_custom_config_path(monkeypatch, tmp_path):
    """`--config-path` should override the auto-detected Claude path."""
    custom_dir = tmp_path / "my-claude"
    monkeypatch.setattr(sys, "argv", [
        "__main__.py", "--api-key", "k", "--config-path", str(custom_dir)
    ])

    runpy = pytest.importorskip("runpy")
    runpy.run_module(
        "minimax_mcp.__main__", run_name="__main__", alter_sys=True
    )

    out_file = custom_dir / "claude_desktop_config.json"
    assert out_file.exists()
    parsed = json.loads(out_file.read_text())
    assert parsed["mcpServers"]["Minimax"]["env"]["MINIMAX_API_KEY"] == "k"


def test_main_block_missing_config_path_exits(monkeypatch, capsys):
    """On a non-existent auto-detected path with no --config-path, the script should exit 1."""
    # Point home at an empty tmp dir and clear any platform-specific config dirs
    monkeypatch.setattr(sys, "argv", ["__main__.py", "--api-key", "k"])
    monkeypatch.setattr(sys, "platform", "plan9")  # not handled -> returns None
    # The module loads os.environ at import time, so we patch get_claude_config_path directly
    monkeypatch.setattr(mcp_main, "get_claude_config_path", lambda: None)

    runpy = pytest.importorskip("runpy")
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module(
            "minimax_mcp.__main__", run_name="__main__", alter_sys=True
        )

    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Could not find Claude config path" in captured.out
