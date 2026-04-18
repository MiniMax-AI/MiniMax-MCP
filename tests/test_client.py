import pytest
from unittest.mock import patch, MagicMock
from minimax_mcp.client import MinimaxAPIClient
from minimax_mcp.exceptions import MinimaxAuthError, MinimaxRequestError
import requests


@pytest.fixture
def client():
    return MinimaxAPIClient("test-key", "https://api.test.com")


def test_client_sets_auth_header(client):
    assert client.session.headers["Authorization"] == "Bearer test-key"
    assert client.session.headers["MM-API-Source"] == "Minimax-MCP"


def test_client_raises_auth_error_on_1004(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"Trace-Id": "abc"}
    mock_resp.json.return_value = {
        "base_resp": {"status_code": 1004, "status_msg": "invalid api key"}
    }
    with patch.object(client.session, "request", return_value=mock_resp):
        with pytest.raises(MinimaxAuthError):
            client.get("/v1/test")


def test_client_raises_request_error_on_api_error(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"Trace-Id": "abc"}
    mock_resp.json.return_value = {
        "base_resp": {"status_code": 9999, "status_msg": "unknown error"}
    }
    with patch.object(client.session, "request", return_value=mock_resp):
        with pytest.raises(MinimaxRequestError):
            client.post("/v1/test", json={})


def test_client_raises_on_network_error(client):
    with patch.object(
        client.session, "request", side_effect=requests.exceptions.ConnectionError("fail")
    ):
        with pytest.raises(MinimaxRequestError, match="Request failed"):
            client.get("/v1/test")


def test_client_returns_data_on_success(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"Trace-Id": "abc"}
    mock_resp.json.return_value = {
        "base_resp": {"status_code": 0, "status_msg": "ok"},
        "data": {"result": "test"},
    }
    with patch.object(client.session, "request", return_value=mock_resp):
        result = client.get("/v1/test")
        assert result["data"]["result"] == "test"


def test_client_request_has_timeout(client):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status = MagicMock()
    mock_resp.headers = {"Trace-Id": "abc"}
    mock_resp.json.return_value = {"base_resp": {"status_code": 0, "status_msg": "ok"}}
    with patch.object(client.session, "request", return_value=mock_resp) as mock_req:
        client.get("/v1/test")
        _, kwargs = mock_req.call_args
        assert "timeout" in kwargs
