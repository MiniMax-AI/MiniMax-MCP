"""Tests for Minimax MCP Server tools."""

import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock environment variables BEFORE importing server module
os.environ["MINIMAX_API_KEY"] = "test_api_key"
os.environ["MINIMAX_API_HOST"] = "https://api.minimax.io"
os.environ["MINIMAX_API_RESOURCE_MODE"] = "local"
os.environ["MINIMAX_MCP_BASE_PATH"] = "/tmp"
os.environ["FASTMCP_LOG_LEVEL"] = "WARNING"

from mcp.types import TextContent
from minimax_mcp.exceptions import MinimaxAPIError, MinimaxRequestError

# Import server module functions after env vars are set
from minimax_mcp.server import (
    text_to_audio,
    list_voices,
    voice_clone,
    play_audio,
    generate_video,
    query_video_generation,
    text_to_image,
    music_generation,
    voice_design,
)


# Sample test data
SAMPLE_AUDIO_HEX = "ffd8ffe000104a46494600010100000100" * 100
SAMPLE_TASK_ID = "test_task_123"
SAMPLE_FILE_ID = "test_file_456"
SAMPLE_DOWNLOAD_URL = "https://example.com/video.mp4"
SAMPLE_IMAGE_URL = "https://example.com/image.jpg"


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    return Mock()


@pytest.fixture
def mock_requests_get():
    """Mock requests.get for URL-based operations."""
    with patch("minimax_mcp.server.requests.get") as mock_get:
        yield mock_get


class TestTextToAudio:
    """Tests for text_to_audio tool."""

    def test_text_to_audio_success_with_hex_audio(self, mock_api_client, temp_dir):
        """Test successful audio generation with hex audio data."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "data": {"audio": SAMPLE_AUDIO_HEX}
            }
            
            result = text_to_audio(
                text="Hello world",
                output_directory=str(temp_dir),
                voice_id="male-qn-qingse"
            )
            
            assert isinstance(result, TextContent)
            assert "Success" in result.text
            assert "Voice used: male-qn-qingse" in result.text

    def test_text_to_audio_success_with_url_mode(self, mock_api_client):
        """Test successful audio generation in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                mock_api_client.post.return_value = {
                    "data": {"audio": "https://example.com/audio.mp3"}
                }
                
                result = text_to_audio(text="Hello world")
                
                assert isinstance(result, TextContent)
                assert "Audio URL:" in result.text

    def test_text_to_audio_empty_text_raises_error(self):
        """Test that empty text raises MinimaxRequestError."""
        with pytest.raises(MinimaxRequestError, match="Text is required"):
            text_to_audio(text="")

    def test_text_to_audio_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message in TextContent."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = text_to_audio(text="Hello world")
            
            assert isinstance(result, TextContent)
            assert "Failed to generate audio" in result.text

    def test_text_to_audio_missing_audio_data_returns_error(self, mock_api_client):
        """Test that missing audio data returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {"data": {}}
            
            result = text_to_audio(text="Hello world")
            
            assert isinstance(result, TextContent)
            assert "Failed to get audio data" in result.text


class TestListVoices:
    """Tests for list_voices tool."""

    def test_list_voices_success(self, mock_api_client):
        """Test successful voice listing."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "system_voice": [
                    {"voice_name": "Voice 1", "voice_id": "voice_1"},
                    {"voice_name": "Voice 2", "voice_id": "voice_2"}
                ],
                "voice_cloning": [
                    {"voice_name": "Clone Voice 1", "voice_id": "clone_1"}
                ]
            }
            
            result = list_voices(voice_type="all")
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "Voice 1" in result.text
        assert "voice_1" in result.text
        assert "Clone Voice 1" in result.text

    def test_list_voices_empty_response(self, mock_api_client):
        """Test listing voices with empty response."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {}
            
            result = list_voices()
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text

    def test_list_voices_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = list_voices()
            
            assert isinstance(result, TextContent)
            assert "Failed to list voices" in result.text


class TestVoiceClone:
    """Tests for voice_clone tool."""

    def test_voice_clone_success_with_url(self, mock_api_client, temp_dir, mock_requests_get):
        """Test successful voice cloning from URL."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                # Mock upload response
                mock_api_client.post.side_effect = [
                    {"file": {"file_id": "uploaded_file_123"}},  # Upload response
                    {"demo_audio": "https://example.com/demo.wav"}  # Clone response
                ]
                
                mock_response = Mock()
                mock_response.content = b"fake_audio_data"
                mock_requests_get.return_value = mock_response
                
                result = voice_clone(
                    voice_id="new_voice_id",
                    file="https://example.com/source.mp3",
                    text="Test text",
                    is_url=True,
                    output_directory=str(temp_dir)
                )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text or "Voice cloned successfully" in result.text

    def test_voice_clone_local_file_not_found(self, mock_api_client):
        """Test that non-existent local file returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = voice_clone(
                voice_id="new_voice_id",
                file="/nonexistent/path/audio.mp3",
                text="Test text",
                is_url=False
            )
            
            assert isinstance(result, TextContent)
            assert "Local file does not exist" in result.text

    def test_voice_clone_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.os.path.exists", return_value=True):
                with patch("builtins.open", MagicMock()):
                    mock_api_client.post.side_effect = MinimaxAPIError("API Error")
                    
                    result = voice_clone(
                        voice_id="new_voice_id",
                        file="/path/to/audio.mp3",
                        text="Test text"
                    )
            
            assert isinstance(result, TextContent)
            assert "Failed to clone voice" in result.text

    def test_voice_clone_missing_file_id(self, mock_api_client):
        """Test that missing file_id in upload response returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.os.path.exists", return_value=True):
                with patch("builtins.open", MagicMock()):
                    mock_api_client.post.return_value = {"file": {}}
                    
                    result = voice_clone(
                        voice_id="new_voice_id",
                        file="/path/to/audio.mp3",
                        text="Test text"
                    )
            
            assert isinstance(result, TextContent)
            assert "Failed to get file_id" in result.text

    def test_voice_clone_no_demo_audio(self, mock_api_client, temp_dir):
        """Test voice clone success when no demo audio is returned."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.os.path.exists", return_value=True):
                with patch("builtins.open", MagicMock()):
                    mock_api_client.post.side_effect = [
                        {"file": {"file_id": "uploaded_file_123"}},  # Upload response
                        {}  # Clone response with no demo_audio
                    ]
                    
                    result = voice_clone(
                        voice_id="new_voice_id",
                        file="/path/to/audio.mp3",
                        text="Test text",
                        output_directory=str(temp_dir)
                    )
        
        assert isinstance(result, TextContent)
        assert "Voice cloned successfully" in result.text


class TestPlayAudio:
    """Tests for play_audio tool."""

    def test_play_audio_from_url(self, mock_requests_get):
        """Test playing audio from URL."""
        with patch("minimax_mcp.server.play") as mock_play:
            mock_response = Mock()
            mock_response.content = b"fake_audio_data"
            mock_requests_get.return_value = mock_response
            
            result = play_audio(
                input_file_path="https://example.com/audio.mp3",
                is_url=True
            )
        
        assert isinstance(result, TextContent)
        assert "Successfully played audio file" in result.text
        mock_play.assert_called_once_with(b"fake_audio_data")

    def test_play_audio_from_local_file(self):
        """Test playing local audio file."""
        with patch("minimax_mcp.server.process_input_file") as mock_process:
            with patch("minimax_mcp.server.play") as mock_play:
                with patch("builtins.open", MagicMock()):
                    mock_process.return_value = "/path/to/audio.mp3"
                    
                    result = play_audio(
                        input_file_path="/path/to/audio.mp3",
                        is_url=False
                    )
        
        assert isinstance(result, TextContent)
        assert "Successfully played audio file" in result.text
        mock_play.assert_called_once()


class TestGenerateVideo:
    """Tests for generate_video tool."""

    def test_generate_video_success_sync(self, mock_api_client, temp_dir, mock_requests_get):
        """Test successful synchronous video generation."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.time.sleep"):
                # Mock API responses
                mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
                mock_api_client.get.side_effect = [
                    {"status": "Success", "file_id": SAMPLE_FILE_ID},  # Query status
                    {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}    # Retrieve file
                ]
                
                mock_response = Mock()
                mock_response.content = b"fake_video_data"
                mock_response.raise_for_status = Mock()
                mock_requests_get.return_value = mock_response
                
                result = generate_video(
                    model="MiniMax-Hailuo-02",
                    prompt="A beautiful sunset",
                    output_directory=str(temp_dir)
                )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "Video saved as" in result.text

    def test_generate_video_async_mode(self, mock_api_client):
        """Test video generation in async mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
            
            result = generate_video(
                model="MiniMax-Hailuo-02",
                prompt="A beautiful sunset",
                async_mode=True
            )
        
        assert isinstance(result, TextContent)
        assert "Task ID:" in result.text
        assert "query_video_generation" in result.text

    def test_generate_video_empty_prompt_returns_error(self, mock_api_client):
        """Test that empty prompt returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = generate_video(prompt="")
            
            assert isinstance(result, TextContent)
            assert "Prompt is required" in result.text

    def test_generate_video_missing_task_id_returns_error(self, mock_api_client):
        """Test that missing task_id returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {}
            
            result = generate_video(prompt="A beautiful sunset")
            
            assert isinstance(result, TextContent)
            assert "Failed to get task_id" in result.text

    def test_generate_video_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = generate_video(prompt="A beautiful sunset")
            
            assert isinstance(result, TextContent)
            assert "Failed to generate video" in result.text

    def test_generate_video_url_mode(self, mock_api_client, temp_dir, mock_requests_get):
        """Test video generation in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                with patch("minimax_mcp.server.time.sleep"):
                    mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
                    mock_api_client.get.side_effect = [
                        {"status": "Success", "file_id": SAMPLE_FILE_ID},
                        {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}
                    ]
                    
                    result = generate_video(
                        prompt="A beautiful sunset",
                        output_directory=str(temp_dir)
                    )
        
        assert isinstance(result, TextContent)
        assert "Video URL:" in result.text

    def test_generate_video_with_first_frame_image_local(self, mock_api_client, temp_dir, mock_requests_get):
        """Test video generation with local first frame image."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.time.sleep"):
                # Create a temporary image file
                image_path = temp_dir / "first_frame.jpg"
                image_path.write_bytes(b"fake_image_data")
                
                mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
                mock_api_client.get.side_effect = [
                    {"status": "Success", "file_id": SAMPLE_FILE_ID},
                    {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}
                ]
                
                mock_response = Mock()
                mock_response.content = b"fake_video_data"
                mock_response.raise_for_status = Mock()
                mock_requests_get.return_value = mock_response
                
                result = generate_video(
                    model="I2V-01",
                    prompt="Video from image",
                    first_frame_image=str(image_path),
                    output_directory=str(temp_dir)
                )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text

    def test_generate_video_first_frame_image_not_found(self, mock_api_client):
        """Test that non-existent first frame image returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = generate_video(
                model="I2V-01",
                prompt="Video from image",
                first_frame_image="/nonexistent/image.jpg"
            )
            
            assert isinstance(result, TextContent)
            assert "First frame image does not exist" in result.text

    def test_generate_video_status_fail_returns_error(self, mock_api_client):
        """Test that failed video status returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.time.sleep"):
                mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
                mock_api_client.get.return_value = {"status": "Fail"}
                
                result = generate_video(prompt="A beautiful sunset")
                
                assert isinstance(result, TextContent)
                assert "Video generation failed" in result.text


class TestQueryVideoGeneration:
    """Tests for query_video_generation tool."""

    def test_query_video_success_with_download(self, mock_api_client, temp_dir, mock_requests_get):
        """Test successful video query with file download."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.get.side_effect = [
                {"status": "Success", "file_id": SAMPLE_FILE_ID},
                {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}
            ]
            
            mock_response = Mock()
            mock_response.content = b"fake_video_data"
            mock_response.raise_for_status = Mock()
            mock_requests_get.return_value = mock_response
            
            result = query_video_generation(
                task_id=SAMPLE_TASK_ID,
                output_directory=str(temp_dir)
            )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "Video saved as" in result.text

    def test_query_video_still_processing(self, mock_api_client):
        """Test video query when task is still processing."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.get.return_value = {"status": "Processing"}
            
            result = query_video_generation(task_id=SAMPLE_TASK_ID)
        
        assert isinstance(result, TextContent)
        assert "still processing" in result.text

    def test_query_video_failed(self, mock_api_client):
        """Test video query when task has failed."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.get.return_value = {"status": "Fail"}
            
            result = query_video_generation(task_id=SAMPLE_TASK_ID)
        
        assert isinstance(result, TextContent)
        assert "FAILED" in result.text

    def test_query_video_url_mode(self, mock_api_client):
        """Test video query in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                mock_api_client.get.side_effect = [
                    {"status": "Success", "file_id": SAMPLE_FILE_ID},
                    {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}
                ]
                
                result = query_video_generation(task_id=SAMPLE_TASK_ID)
        
        assert isinstance(result, TextContent)
        assert "Video URL:" in result.text

    def test_query_video_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.get.side_effect = MinimaxAPIError("API Error")
            
            result = query_video_generation(task_id=SAMPLE_TASK_ID)
            
            assert isinstance(result, TextContent)
            assert "Failed to query video generation status" in result.text


class TestTextToImage:
    """Tests for text_to_image tool."""

    def test_text_to_image_success(self, mock_api_client, temp_dir, mock_requests_get):
        """Test successful image generation."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "data": {
                    "image_urls": [SAMPLE_IMAGE_URL]
                }
            }
            
            mock_response = Mock()
            mock_response.content = b"fake_image_data"
            mock_response.raise_for_status = Mock()
            mock_requests_get.return_value = mock_response
            
            result = text_to_image(
                prompt="A beautiful flower",
                output_directory=str(temp_dir)
            )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "Images saved as" in result.text

    def test_text_to_image_empty_prompt_returns_error(self, mock_api_client):
        """Test that empty prompt returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = text_to_image(prompt="")
            
            assert isinstance(result, TextContent)
            assert "Prompt is required" in result.text

    def test_text_to_image_no_images_returns_error(self, mock_api_client):
        """Test that missing image URLs returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {"data": {}}
            
            result = text_to_image(prompt="A beautiful flower")
            
            assert isinstance(result, TextContent)
            assert "No images generated" in result.text

    def test_text_to_image_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = text_to_image(prompt="A beautiful flower")
            
            assert isinstance(result, TextContent)
            assert "Failed to generate images" in result.text

    def test_text_to_image_url_mode(self, mock_api_client):
        """Test image generation in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                mock_api_client.post.return_value = {
                    "data": {
                        "image_urls": [SAMPLE_IMAGE_URL]
                    }
                }
                
                result = text_to_image(prompt="A beautiful flower")
        
        assert isinstance(result, TextContent)
        assert "Image URLs:" in result.text

    def test_text_to_image_multiple_images(self, mock_api_client, temp_dir, mock_requests_get):
        """Test image generation with multiple images."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "data": {
                    "image_urls": [
                        "https://example.com/image1.jpg",
                        "https://example.com/image2.jpg"
                    ]
                }
            }
            
            mock_response = Mock()
            mock_response.content = b"fake_image_data"
            mock_response.raise_for_status = Mock()
            mock_requests_get.return_value = mock_response
            
            result = text_to_image(
                prompt="Flowers",
                n=2,
                output_directory=str(temp_dir)
            )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text


class TestMusicGeneration:
    """Tests for music_generation tool."""

    def test_music_generation_success(self, mock_api_client, temp_dir):
        """Test successful music generation."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "data": {"audio": SAMPLE_AUDIO_HEX}
            }
            
            result = music_generation(
                prompt="Upbeat pop music",
                lyrics="Line one\nLine two\nLine three",
                output_directory=str(temp_dir)
            )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "Music saved as" in result.text

    def test_music_generation_url_mode(self, mock_api_client):
        """Test music generation in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                mock_api_client.post.return_value = {
                    "data": {"audio": "https://example.com/music.mp3"}
                }
                
                result = music_generation(
                    prompt="Upbeat pop music",
                    lyrics="Line one\nLine two\nLine three"
                )
        
        assert isinstance(result, TextContent)
        assert "Music url:" in result.text

    def test_music_generation_empty_prompt_returns_error(self, mock_api_client):
        """Test that empty prompt returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = music_generation(prompt="", lyrics="Valid lyrics")
            
            assert isinstance(result, TextContent)
            assert "Prompt is required" in result.text

    def test_music_generation_empty_lyrics_returns_error(self, mock_api_client):
        """Test that empty lyrics returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = music_generation(prompt="Valid prompt", lyrics="")
            
            assert isinstance(result, TextContent)
            assert "Lyrics is required" in result.text

    def test_music_generation_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = music_generation(
                prompt="Upbeat pop music",
                lyrics="Line one\nLine two\nLine three"
            )
            
            assert isinstance(result, TextContent)
            assert "Failed to generate music" in result.text


class TestVoiceDesign:
    """Tests for voice_design tool."""

    def test_voice_design_success(self, mock_api_client, temp_dir):
        """Test successful voice design generation."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "voice_id": "designed_voice_123",
                "trial_audio": SAMPLE_AUDIO_HEX
            }
            
            result = voice_design(
                prompt="Warm and friendly voice",
                preview_text="Hello, how are you?",
                output_directory=str(temp_dir)
            )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
        assert "designed_voice_123" in result.text
        assert "File saved as" in result.text

    def test_voice_design_url_mode(self, mock_api_client):
        """Test voice design in URL mode."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.resource_mode", "url"):
                mock_api_client.post.return_value = {
                    "voice_id": "designed_voice_123",
                    "trial_audio": "https://example.com/audio.mp3"
                }
                
                result = voice_design(
                    prompt="Warm and friendly voice",
                    preview_text="Hello, how are you?"
                )
        
        assert isinstance(result, TextContent)
        assert "Voice ID generated" in result.text
        assert "Trial Audio" in result.text

    def test_voice_design_empty_prompt_returns_error(self, mock_api_client):
        """Test that empty prompt returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = voice_design(prompt="", preview_text="Hello")
            
            assert isinstance(result, TextContent)
            assert "prompt is required" in result.text

    def test_voice_design_empty_preview_text_returns_error(self, mock_api_client):
        """Test that empty preview_text returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            result = voice_design(prompt="Valid prompt", preview_text="")
            
            assert isinstance(result, TextContent)
            assert "preview_text is required" in result.text

    def test_voice_design_no_voice_id_returns_error(self, mock_api_client):
        """Test that missing voice_id returns error."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {}
            
            result = voice_design(
                prompt="Warm and friendly voice",
                preview_text="Hello, how are you?"
            )
            
            assert isinstance(result, TextContent)
            assert "No voice generated" in result.text

    def test_voice_design_api_error_returns_error_message(self, mock_api_client):
        """Test that API error returns error message."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.side_effect = MinimaxAPIError("API Error")
            
            result = voice_design(
                prompt="Warm and friendly voice",
                preview_text="Hello, how are you?"
            )
            
            assert isinstance(result, TextContent)
            assert "Failed to design voice" in result.text


class TestToolInputsValidation:
    """Test validation of tool inputs."""

    def test_text_to_audio_speed_range(self, mock_api_client, temp_dir):
        """Test text_to_audio with speed parameter."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            mock_api_client.post.return_value = {
                "data": {"audio": SAMPLE_AUDIO_HEX}
            }
            
            result = text_to_audio(text="Hello", speed=1.5, output_directory=str(temp_dir))
            assert isinstance(result, TextContent)
            assert "Success" in result.text

    def test_generate_video_with_hailuo_02_model(self, mock_api_client, temp_dir, mock_requests_get):
        """Test video generation with MiniMax-Hailuo-02 model."""
        with patch("minimax_mcp.server.api_client", mock_api_client):
            with patch("minimax_mcp.server.time.sleep"):
                mock_api_client.post.return_value = {"task_id": SAMPLE_TASK_ID}
                mock_api_client.get.side_effect = [
                    {"status": "Success", "file_id": SAMPLE_FILE_ID},
                    {"file": {"download_url": SAMPLE_DOWNLOAD_URL}}
                ]
                
                mock_response = Mock()
                mock_response.content = b"fake_video_data"
                mock_response.raise_for_status = Mock()
                mock_requests_get.return_value = mock_response
                
                result = generate_video(
                    model="MiniMax-Hailuo-02",
                    prompt="A beautiful sunset",
                    duration=10,
                    resolution="1080P",
                    output_directory=str(temp_dir)
                )
        
        assert isinstance(result, TextContent)
        assert "Success" in result.text
