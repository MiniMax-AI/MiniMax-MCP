from minimax_mcp.client import MinimaxAPIClient
from minimax_mcp.exceptions import MinimaxAuthError


class _Response:
    def __init__(self, payload, headers=None):
        self._payload = payload
        self.headers = headers or {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def test_success_without_base_resp(monkeypatch):
    client = MinimaxAPIClient("test-key", "https://api.example.invalid")

    def fake_request(method, url, **kwargs):
        assert url.endswith("/v1/images/generations")
        return _Response({"data": [{"url": "https://example.invalid/h3.jpg"}]})

    monkeypatch.setattr(client.session, "request", fake_request)
    data = client.post("/v1/images/generations", json={"model": "MiniMax-H3-Image"})
    assert data["data"][0]["url"] == "https://example.invalid/h3.jpg"


def test_base_resp_error_still_raises(monkeypatch):
    client = MinimaxAPIClient("test-key", "https://api.example.invalid")

    def fake_request(method, url, **kwargs):
        return _Response(
            {"base_resp": {"status_code": 1004, "status_msg": "auth failed"}},
            headers={"Trace-Id": "trace-1"},
        )

    monkeypatch.setattr(client.session, "request", fake_request)
    try:
        client.post("/v1/image_generation", json={})
    except MinimaxAuthError as exc:
        assert "auth failed" in str(exc)
    else:
        raise AssertionError("expected MinimaxAuthError")
