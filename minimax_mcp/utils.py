import os
from pathlib import Path
from datetime import datetime
from fuzzywuzzy import fuzz
import shutil
import subprocess
import base64
import requests
from typing import Iterator, Union
from minimax_mcp.const import *
from minimax_mcp.exceptions import MinimaxMcpError, MinimaxRequestError


def is_file_writeable(path: Path) -> bool:
    if path.exists():
        return os.access(path, os.W_OK)
    parent_dir = path.parent
    return os.access(parent_dir, os.W_OK)


def build_output_file(
    tool: str, text: str, output_path: Path, extension: str, full_id: bool = False
) -> Path:
    id = text if full_id else text[:10]

    output_file_name = f"{tool}_{id.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{extension}"
    return output_path / output_file_name


def build_output_path(
    output_directory: str | None, base_path: str | None = None, is_test: bool = False
) -> Path:
    # Set default base_path to desktop if not provided
    if base_path is None:
        base_path = str(Path.home() / "Desktop")
    
    # Handle output path based on output_directory
    if output_directory is None:
        output_path = Path(os.path.expanduser(base_path))
    elif not os.path.isabs(os.path.expanduser(output_directory)):
        output_path = Path(os.path.expanduser(base_path)) / Path(output_directory)
    else:
        output_path = Path(os.path.expanduser(output_directory))

    # Safety checks and directory creation
    if is_test:
        return output_path
    if not is_file_writeable(output_path):
        raise MinimaxMcpError(f"Directory ({output_path}) is not writeable")
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def find_similar_filenames(
    target_file: str, directory: Path, threshold: int = 70
) -> list[tuple[str, int]]:
    """
    Find files with names similar to the target file using fuzzy matching.

    Args:
        target_file (str): The reference filename to compare against
        directory (str): Directory to search in (defaults to current directory)
        threshold (int): Similarity threshold (0 to 100, where 100 is identical)

    Returns:
        list: List of similar filenames with their similarity scores
    """
    target_filename = os.path.basename(target_file)
    similar_files = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if (
                filename == target_filename
                and os.path.join(root, filename) == target_file
            ):
                continue
            similarity = fuzz.token_sort_ratio(target_filename, filename)

            if similarity >= threshold:
                file_path = Path(root) / filename
                similar_files.append((file_path, similarity))

    similar_files.sort(key=lambda x: x[1], reverse=True)

    return similar_files


def try_find_similar_files(
    filename: str, directory: Path, take_n: int = 5
) -> list[Path]:
    similar_files = find_similar_filenames(filename, directory)
    if not similar_files:
        return []

    filtered_files = []

    for path, _ in similar_files[:take_n]:
        if check_audio_file(path):
            filtered_files.append(path)

    return filtered_files


def check_audio_file(path: Path) -> bool:
    audio_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".flac",
        ".mp4",
        ".avi",
        ".mov",
        ".wmv",
    }
    return path.suffix.lower() in audio_extensions


def process_input_file(file_path: str, audio_content_check: bool = True) -> Path:
    if not os.path.isabs(file_path) and not os.environ.get(ENV_MINIMAX_MCP_BASE_PATH):
        raise MinimaxMcpError(
            "File path must be an absolute path if MINIMAX_MCP_BASE_PATH is not set"
        )
    path = Path(file_path)
    if not path.exists() and path.parent.exists():
        parent_directory = path.parent
        similar_files = try_find_similar_files(path.name, parent_directory)
        similar_files_formatted = ",".join([str(file) for file in similar_files])
        if similar_files:
            raise MinimaxMcpError(
                f"File ({path}) does not exist. Did you mean any of these files: {similar_files_formatted}?"
            )
        raise MinimaxMcpError(f"File ({path}) does not exist")
    elif not path.exists():
        raise MinimaxMcpError(f"File ({path}) does not exist")
    elif not path.is_file():
        raise MinimaxMcpError(f"File ({path}) is not a file")

    if audio_content_check and not check_audio_file(path):
        raise MinimaxMcpError(f"File ({path}) is not an audio or video file")
    return path


def is_installed(lib_name: str) -> bool:
    return shutil.which(lib_name) is not None


def play(
    audio: Union[bytes, Iterator[bytes]]
) -> None:
    if isinstance(audio, Iterator):
        audio = b"".join(audio)

    if not is_installed("ffplay"):
        message = (
            "ffplay from ffmpeg not found, necessary to play audio. "
            "mac: install it with 'brew install ffmpeg'. "
            "linux or windows: install it from https://ffmpeg.org/"
        )
        raise ValueError(message)
    
    args = ["ffplay", "-autoexit", "-", "-nodisp"]
    proc = subprocess.Popen(
        args=args,
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = proc.communicate(input=audio)

    proc.poll()


def process_image_url(image_url: str) -> str:
    """
    Process image URL and convert to base64 data URL format.
    
    This function handles three types of image inputs:
    1. HTTP/HTTPS URLs: Downloads the image and converts to base64
    2. Base64 data URLs: Passes through as-is
    3. Local file paths: Reads the file and converts to base64
    
    Args:
        image_url (str): The image URL, data URL, or local file path
        
    Returns:
        str: Base64 data URL in format "data:image/{format};base64,{data}"
        
    Raises:
        MinimaxRequestError: If image cannot be downloaded, read, or processed
    """
    # If already in base64 data URL format, pass through
    if image_url.startswith("data:"):
        return image_url
    
    # Handle HTTP/HTTPS URLs
    if image_url.startswith(("http://", "https://")):
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_data = image_response.content
            
            # Detect image format from content-type header
            content_type = image_response.headers.get('content-type', '').lower()
            if 'jpeg' in content_type or 'jpg' in content_type:
                image_format = 'jpeg'
            elif 'png' in content_type:
                image_format = 'png'
            elif 'gif' in content_type:
                image_format = 'gif'
            elif 'webp' in content_type:
                image_format = 'webp'
            else:
                # Default to jpeg if cannot detect
                image_format = 'jpeg'
            
            # Convert to base64 data URL
            base64_data = base64.b64encode(image_data).decode('utf-8')
            return f"data:image/{image_format};base64,{base64_data}"
            
        except requests.RequestException as e:
            raise MinimaxRequestError(f"Failed to download image from URL: {str(e)}")
    
    # Handle local file paths
    else:
        if not os.path.exists(image_url):
            raise MinimaxRequestError(f"Local image file does not exist: {image_url}")
        
        try:
            with open(image_url, "rb") as f:
                image_data = f.read()
                
                # Detect image format from file extension
                image_format = 'jpeg'  # Default
                if image_url.lower().endswith('.png'):
                    image_format = 'png'
                elif image_url.lower().endswith('.gif'):
                    image_format = 'gif'
                elif image_url.lower().endswith('.webp'):
                    image_format = 'webp'
                elif image_url.lower().endswith(('.jpg', '.jpeg')):
                    image_format = 'jpeg'
                
                base64_data = base64.b64encode(image_data).decode('utf-8')
                return f"data:image/{image_format};base64,{base64_data}"
                
        except IOError as e:
            raise MinimaxRequestError(f"Failed to read local image file: {str(e)}")


