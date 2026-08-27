import pytest
from unittest.mock import patch, MagicMock
import requests
from minimax_mcp.client import MinimaxAPIClient
from minimax_mcp.exceptions import MinimaxAuthError, MinimaxRequestError


def test_initialization(api_client):
    assert api_client.api_key == "test-key"
    assert api_client.api_host == "https://api.test.com"
    assert api_client.session.headers["Authorization"] == "Bearer test-key"
    assert api_client.session.headers["MM-API-Source"] == "Minimax-MCP"


def test_successful_json_post(api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 0},
        "data": {"result": "ok"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response):
        result = api_client.post("/v1/test", json={"key": "value"})

    assert result["data"]["result"] == "ok"
    assert api_client.session.headers["Content-Type"] == "application/json"


def test_successful_get(api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 0},
        "status": "Success",
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response) as mock_req:
        result = api_client.get("/v1/query?task_id=123")

    mock_req.assert_called_once_with("GET", "https://api.test.com/v1/query?task_id=123")
    assert result["status"] == "Success"


def test_file_upload_removes_content_type(api_client):
    api_client.session.headers["Content-Type"] = "application/json"

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 0},
        "file": {"file_id": "abc"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response):
        result = api_client.post("/v1/files/upload", files={"file": b"data"}, data={"purpose": "voice_clone"})

    assert "Content-Type" not in api_client.session.headers
    assert result["file"]["file_id"] == "abc"


def test_auth_error_1004(api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 1004, "status_msg": "invalid api key"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response):
        with pytest.raises(MinimaxAuthError, match="API key"):
            api_client.post("/v1/test", json={})


def test_verification_error_2038(api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 2038, "status_msg": "need verification"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response):
        with pytest.raises(MinimaxRequestError, match="real-name verification"):
            api_client.post("/v1/test", json={})


def test_generic_api_error(api_client):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "base_resp": {"status_code": 9999, "status_msg": "unknown error"},
    }
    mock_response.raise_for_status = MagicMock()
    mock_response.headers = {"Trace-Id": "abc123"}

    with patch.object(api_client.session, "request", return_value=mock_response):
        with pytest.raises(MinimaxRequestError, match="9999"):
            api_client.post("/v1/test", json={})


def test_network_request_failure(api_client):
    with patch.object(
        api_client.session, "request", side_effect=requests.exceptions.ConnectionError("Connection refused")
    ):
        with pytest.raises(MinimaxRequestError, match="Request failed"):
            api_client.get("/v1/test")
