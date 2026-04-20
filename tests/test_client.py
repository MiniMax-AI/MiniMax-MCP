"""Tests for MinimaxAPIClient."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import requests

from minimax_mcp.client import MinimaxAPIClient
from minimax_mcp.exceptions import MinimaxAuthError, MinimaxRequestError


@pytest.fixture
def api_client():
    """Create an API client instance for testing."""
    return MinimaxAPIClient(api_key="test_api_key", api_host="https://api.minimax.io")


class TestMinimaxAPIClientInit:
    """Tests for MinimaxAPIClient initialization."""

    def test_init_sets_api_key_and_host(self, api_client):
        assert api_client.api_key == "test_api_key"
        assert api_client.api_host == "https://api.minimax.io"

    def test_init_creates_session(self, api_client):
        assert isinstance(api_client.session, requests.Session)

    def test_init_sets_authorization_header(self, api_client):
        assert "Authorization" in api_client.session.headers
        assert api_client.session.headers["Authorization"] == "Bearer test_api_key"

    def test_init_sets_source_header(self, api_client):
        assert "MM-API-Source" in api_client.session.headers
        assert api_client.session.headers["MM-API-Source"] == "Minimax-MCP"


class TestMakeRequest:
    """Tests for _make_request method."""

    def test_make_request_get_sets_content_type_json(self, api_client):
        """GET requests should set Content-Type to application/json."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {"base_resp": {"status_code": 0}}
            mock_response.headers = {}
            mock_request.return_value = mock_response

            api_client._make_request("GET", "/test_endpoint")

            # Check that Content-Type was set for non-file requests
            assert api_client.session.headers.get("Content-Type") == "application/json"

    def test_make_request_post_with_files_removes_content_type(self, api_client):
        """POST with files should not set Content-Type (requests handles it)."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {"base_resp": {"status_code": 0}}
            mock_response.headers = {}
            mock_request.return_value = mock_response

            # Simulate files being passed
            api_client.session.headers["Content-Type"] = "application/json"
            api_client._make_request("POST", "/test", files={"file": "data"})

            # Content-Type should be removed for multipart
            assert "Content-Type" not in api_client.session.headers

    def test_make_request_success_returns_data(self, api_client):
        """Successful request should return parsed JSON data."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {
                "base_resp": {"status_code": 0},
                "data": {"key": "value"}
            }
            mock_response.headers = {}
            mock_request.return_value = mock_response

            result = api_client._make_request("GET", "/test_endpoint")

            assert result == {"base_resp": {"status_code": 0}, "data": {"key": "value"}}
            mock_request.assert_called_once()

    def test_make_request_raises_for_status_4xx(self, api_client):
        """HTTP 4xx errors should raise RequestException."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Client Error")
            mock_response.headers = {}
            mock_request.return_value = mock_response

            with pytest.raises(MinimaxRequestError, match="Request failed"):
                api_client._make_request("GET", "/test_endpoint")

    def test_make_request_raises_for_status_5xx(self, api_client):
        """HTTP 5xx errors should raise RequestException."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.status_code = 500
            mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
            mock_response.headers = {}
            mock_request.return_value = mock_response

            with pytest.raises(MinimaxRequestError, match="Request failed"):
                api_client._make_request("GET", "/test_endpoint")

    def test_make_request_api_error_1004_raises_auth_error(self, api_client):
        """API status_code 1004 should raise MinimaxAuthError."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {
                "base_resp": {"status_code": 1004, "status_msg": "Invalid API key"}
            }
            mock_response.headers = {"Trace-Id": "trace-123"}
            mock_request.return_value = mock_response

            with pytest.raises(MinimaxAuthError) as exc_info:
                api_client._make_request("GET", "/test_endpoint")

            assert "Invalid API key" in str(exc_info.value)
            assert "trace-123" in str(exc_info.value)

    def test_make_request_api_error_2038_raises_request_error(self, api_client):
        """API status_code 2038 (real-name verification) should raise MinimaxRequestError."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {
                "base_resp": {"status_code": 2038, "status_msg": "Real-name verification required"}
            }
            mock_response.headers = {"Trace-Id": "trace-456"}
            mock_request.return_value = mock_response

            with pytest.raises(MinimaxRequestError) as exc_info:
                api_client._make_request("GET", "/test_endpoint")

            assert "Real-name verification required" in str(exc_info.value)
            assert "platform.minimaxi.com" in str(exc_info.value)

    def test_make_request_api_error_other_code(self, api_client):
        """Other non-zero status codes should raise MinimaxRequestError."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_response = Mock()
            mock_response.json.return_value = {
                "base_resp": {"status_code": 9999, "status_msg": "Unknown error"}
            }
            mock_response.headers = {"Trace-Id": "trace-789"}
            mock_request.return_value = mock_response

            with pytest.raises(MinimaxRequestError) as exc_info:
                api_client._make_request("GET", "/test_endpoint")

            assert "9999" in str(exc_info.value)
            assert "Unknown error" in str(exc_info.value)

    def test_make_request_connection_error(self, api_client):
        """Connection errors should raise MinimaxRequestError."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_request.side_effect = requests.exceptions.ConnectionError("Connection refused")

            with pytest.raises(MinimaxRequestError, match="Request failed"):
                api_client._make_request("GET", "/test_endpoint")

    def test_make_request_timeout_error(self, api_client):
        """Timeout errors should raise MinimaxRequestError."""
        with patch.object(api_client.session, "request") as mock_request:
            mock_request.side_effect = requests.exceptions.Timeout("Request timed out")

            with pytest.raises(MinimaxRequestError, match="Request failed"):
                api_client._make_request("GET", "/test_endpoint")


class TestGetMethod:
    """Tests for get method."""

    def test_get_calls_make_request_with_get(self, api_client):
        """get() should call _make_request with GET method."""
        with patch.object(api_client, "_make_request") as mock_make_request:
            mock_make_request.return_value = {"data": "test"}
            
            result = api_client.get("/test_endpoint", param="value")

            mock_make_request.assert_called_once_with("GET", "/test_endpoint", param="value")
            assert result == {"data": "test"}


class TestPostMethod:
    """Tests for post method."""

    def test_post_calls_make_request_with_post(self, api_client):
        """post() should call _make_request with POST method."""
        with patch.object(api_client, "_make_request") as mock_make_request:
            mock_make_request.return_value = {"data": "test"}
            
            result = api_client.post("/test_endpoint", json={"key": "value"})

            mock_make_request.assert_called_once_with("POST", "/test_endpoint", json={"key": "value"})
            assert result == {"data": "test"}
