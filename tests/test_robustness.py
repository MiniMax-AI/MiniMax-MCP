import importlib
import os

os.environ.setdefault("MINIMAX_API_KEY", "test-api-key")
os.environ.setdefault("MINIMAX_API_HOST", "https://api.example.invalid")

from minimax_mcp.client import MinimaxAPIClient
from minimax_mcp.const import REQUEST_TIMEOUT


def test_client_requests_carry_timeout():
    """Every API request must send a timeout so a stalled endpoint cannot
    hang the synchronous MCP tool forever."""
    client = MinimaxAPIClient("test-api-key", "https://api.example.invalid")
    seen = {}
    original = client.session.request

    def capture(method, url, **kwargs):
        seen.update(kwargs)
        return original(method, url, **kwargs)

    client.session.request = capture
    try:
        client.post("/v1/get_voice", json={"voice_type": "all"})
    except Exception:
        pass  # the endpoint is unreachable; only the request shape matters
    assert seen.get("timeout") == REQUEST_TIMEOUT


def test_client_timeout_is_overridable_per_call():
    client = MinimaxAPIClient("test-api-key", "https://api.example.invalid")
    seen = {}
    original = client.session.request

    def capture(method, url, **kwargs):
        seen.update(kwargs)
        return original(method, url, **kwargs)

    client.session.request = capture
    try:
        client.get("/v1/files/retrieve?file_id=x", timeout=(5, 30))
    except Exception:
        pass
    assert seen.get("timeout") == (5, 30)


def test_server_entrypoint_writes_nothing_to_stdout(monkeypatch, capsys):
    """stdout is the MCP JSON-RPC transport under stdio; a stray startup
    print corrupts the protocol stream for strict clients."""
    server = importlib.import_module("minimax_mcp.server")
    monkeypatch.setattr(server.mcp, "run", lambda *args, **kwargs: None)
    server.main()
    captured = capsys.readouterr()
    assert captured.out == "", f"stdout polluted: {captured.out!r}"
