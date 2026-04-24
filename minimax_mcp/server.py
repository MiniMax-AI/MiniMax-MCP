"""
MiniMax MCP Server

⚠️ IMPORTANT: This server connects to Minimax API endpoints which may involve costs.
Any tool that makes an API call is clearly marked with a cost warning. Please follow these guidelines:

1. Only use these tools when users specifically ask for them
2. For audio generation tools, be mindful that text length affects the cost
3. Voice cloning features are charged upon first use after cloning

Note: Tools without cost warnings are free to use as they only read existing data.
"""

import os
import base64
import requests
import time
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent, ToolAnnotations
from minimax_mcp.utils import (
    build_output_path,
    build_output_file,
    process_input_file,
    play
)
from pathlib import Path

from minimax_mcp.const import *
from minimax_mcp.exceptions import MinimaxAPIError, MinimaxRequestError
from minimax_mcp.client import MinimaxAPIClient

load_dotenv()
api_key = os.getenv(ENV_MINIMAX_API_KEY)
base_path = os.getenv(ENV_MINIMAX_MCP_BASE_PATH) or "~/Desktop"
api_host = os.getenv(ENV_MINIMAX_API_HOST)
resource_mode = os.getenv(ENV_RESOURCE_MODE) or RESOURCE_MODE_URL
fastmcp_log_level = os.getenv(ENV_FASTMCP_LOG_LEVEL) or "WARNING"

if not api_key:
    raise ValueError("MINIMAX_API_KEY environment variable is required")
if not api_host:
    raise ValueError("MINIMAX_API_HOST environment variable is required")

mcp = FastMCP("Minimax",log_level=fastmcp_log_level)
api_client = MinimaxAPIClient(api_key, api_host)


@mcp.tool(
    annotations=ToolAnnotations(
        title="Text to Audio",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Convert text to audio with a given voice and save the output audio file to a given directory.
    Directory is optional, if not provided, the output file will be saved to $HOME/Desktop.
    Voice id is optional, if not provided, the default voice will be used.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

    Args:
        text (str): The text to convert to speech.
        voice_id (str, optional): The id of the voice to use. For example, "male-qn-qingse"/"audiobook_female_1"/"cute_boy"/"Charming_Lady"...
        model (string, optional): The model to use.
        speed (float, optional): Speed of the generated audio. Controls the speed of the generated speech. Values range from 0.5 to 2.0, with 1.0 being the default speed. 
        vol (float, optional): Volume of the generated audio. Controls the volume of the generated speech. Values range from 0 to 10, with 1 being the default volume.
        pitch (int, optional): Pitch of the generated audio. Controls the speed of the generated speech. Values range from -12 to 12, with 0 being the default speed.
        emotion (str, optional): Emotion of the generated audio. Controls the emotion of the generated speech. Values range ["happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"], with "happy" being the default emotion.
        sample_rate (int, optional): Sample rate of the generated audio. Controls the sample rate of the generated speech. Values range [8000,16000,22050,24000,32000,44100] with 32000 being the default sample rate.
        bitrate (int, optional): Bitrate of the generated audio. Controls the bitrate of the generated speech. Values range [32000,64000,128000,256000] with 128000 being the default bitrate.
        channel (int, optional): Channel of the generated audio. Controls the channel of the generated speech. Values range [1, 2] with 1 being the default channel.
        format (str, optional): Format of the generated audio. Controls the format of the generated speech. Values range ["pcm", "mp3","flac"] with "mp3" being the default format.
        language_boost (str, optional): Language boost of the generated audio. Controls the language boost of the generated speech. Values range ['Chinese', 'Chinese,Yue', 'English', 'Arabic', 'Russian', 'Spanish', 'French', 'Portuguese', 'German', 'Turkish', 'Dutch', 'Ukrainian', 'Vietnamese', 'Indonesian', 'Japanese', 'Italian', 'Korean', 'Thai', 'Polish', 'Romanian', 'Greek', 'Czech', 'Finnish', 'Hindi', 'auto'] with "auto" being the default language boost.
        output_directory (str): The directory to save the audio to.

    Returns:
        Text content with the path to the output file and name of the voice used.
    """
)
def text_to_audio(
    text: str,
    output_directory: str = None,
    voice_id: str = DEFAULT_VOICE_ID,
    model: str = DEFAULT_SPEECH_MODEL,
    speed: float = DEFAULT_SPEED,
    vol: float = DEFAULT_VOLUME,
    pitch: int = DEFAULT_PITCH,
    emotion: str = DEFAULT_EMOTION,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: int = DEFAULT_BITRATE,
    channel: int = DEFAULT_CHANNEL,
    format: str = DEFAULT_FORMAT,
    language_boost: str = DEFAULT_LANGUAGE_BOOST,
):
    if not text:
        raise MinimaxRequestError("Text is required.")

    payload = {
        "model": model,
        "text": text,
        "voice_setting": {
            "voice_id": voice_id,
            "speed": speed,
            "vol": vol,
            "pitch": pitch,
            "emotion": emotion
        },
        "audio_setting": {
            "sample_rate": sample_rate,
            "bitrate": bitrate,
            "format": format,
            "channel": channel
        },
        "language_boost": language_boost
    }
    if resource_mode == RESOURCE_MODE_URL:
        payload["output_format"] = "url"
    try:
        response_data = api_client.post("/v1/t2a_v2", json=payload)
        audio_data = response_data.get('data', {}).get('audio', '')
        
        if not audio_data:
            raise MinimaxRequestError(f"Failed to get audio data from response")
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Audio URL: {audio_data}"
            )
        # hex->bytes
        audio_bytes = bytes.fromhex(audio_data)

        # save audio to file
        output_path = build_output_path(output_directory, base_path)
        output_file_name = build_output_file("t2a", text, output_path, format)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / output_file_name, "wb") as f:
            f.write(audio_bytes)

        return TextContent(
            type="text",
            text=f"Success. File saved as: {output_path / output_file_name}. Voice used: {voice_id}",
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to generate audio: {str(e)}"
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="List Voices",
        readOnlyHint=True,
        openWorldHint=True,
    ),
    description="""List all voices available.

    Args:
        voice_type (str, optional): The type of voices to list. Values range ["all", "system", "voice_cloning"], with "all" being the default.
    Returns:
        Text content with the list of voices.
    """
)
def list_voices(
    voice_type: str = "all"
):
    try:
        response_data = api_client.post("/v1/get_voice", json={'voice_type': voice_type})
        
        system_voices = response_data.get('system_voice', []) or []
        voice_cloning_voices = response_data.get('voice_cloning', []) or []
        system_voice_list = []
        voice_cloning_voice_list = []
        
        for voice in system_voices:
            system_voice_list.append(f"Name: {voice.get('voice_name')}, ID: {voice.get('voice_id')}")
        for voice in voice_cloning_voices:
            voice_cloning_voice_list.append(f"Name: {voice.get('voice_name')}, ID: {voice.get('voice_id')}")

        return TextContent(
            type="text",
            text=f"Success. System Voices: {system_voice_list}, Voice Cloning Voices: {voice_cloning_voice_list}"
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to list voices: {str(e)}"
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Voice Clone",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Clone a voice using provided audio files. The new voice will be charged upon first use.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

     Args:
        voice_id (str): The id of the voice to use.
        file (str): The path to the audio file to clone or a URL to the audio file.
        text (str, optional): The text to use for the demo audio.
        is_url (bool, optional): Whether the file is a URL. Defaults to False.
        output_directory (str): The directory to save the demo audio to.
    Returns:
        Text content with the voice id of the cloned voice.
    """
)
def voice_clone(
    voice_id: str, 
    file: str,
    text: str,
    output_directory: str = None,
    is_url: bool = False
) -> TextContent:
    try:
        # step1: upload file
        if is_url:
            # download file from url
            response = requests.get(file, stream=True)
            response.raise_for_status()
            files = {'file': ('audio_file.mp3', response.raw, 'audio/mpeg')}
            data = {'purpose': 'voice_clone'}
            response_data = api_client.post("/v1/files/upload", files=files, data=data)
        else:
            # open and upload file
            if not os.path.exists(file):
                raise MinimaxRequestError(f"Local file does not exist: {file}")
            with open(file, 'rb') as f:
                files = {'file': f}
                data = {'purpose': 'voice_clone'}
                response_data = api_client.post("/v1/files/upload", files=files, data=data)
            
        file_id = response_data.get("file",{}).get("file_id")
        if not file_id:
            raise MinimaxRequestError(f"Failed to get file_id from upload response")

        # step2: clone voice
        payload = {
            "file_id": file_id,
            "voice_id": voice_id,
        }
        if text:
            payload["text"] = text
            payload["model"] = DEFAULT_SPEECH_MODEL

        response_data = api_client.post("/v1/voice_clone", json=payload)
        
        if not response_data.get("demo_audio"):
            return TextContent(
                type="text",
                text=f"Voice cloned successfully: Voice ID: {voice_id}"
            )
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Demo audio URL: {response_data.get('demo_audio')}"
            )
        # step3: download demo audio
        output_path = build_output_path(output_directory, base_path)
        output_file_name = build_output_file("voice_clone", text, output_path, "wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / output_file_name, "wb") as f:
            f.write(requests.get(response_data.get("demo_audio")).content)

        return TextContent(
            type="text",
            text=f"Voice cloned successfully: Voice ID: {voice_id}, demo audio saved as: {output_path / output_file_name}"
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to clone voice: {str(e)}"
        )
    except (IOError, requests.RequestException) as e:
        return TextContent(
            type="text",
            text=f"Failed to handle files: {str(e)}"
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Play Audio",
        readOnlyHint=True,
        openWorldHint=False,
    ),
    description="""Play an audio file. Supports WAV and MP3 formats. Not supports video.

     Args:
        input_file_path (str): The path to the audio file to play.
        is_url (bool, optional): Whether the audio file is a URL.
    Returns:
        Text content with the path to the audio file.
    """
)
def play_audio(input_file_path: str, is_url: bool = False) -> TextContent:
    if is_url:
        play(requests.get(input_file_path).content)
        return TextContent(type="text", text=f"Successfully played audio file: {input_file_path}")
    else:
        file_path = process_input_file(input_file_path)
        play(open(file_path, "rb").read())
        return TextContent(type="text", text=f"Successfully played audio file: {file_path}")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Generate Video",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Generate a video from a prompt.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

     Args:
        model (str, optional): The model to use. Values range ["T2V-01", "T2V-01-Director", "I2V-01", "I2V-01-Director", "I2V-01-live", "MiniMax-Hailuo-02"]. "Director" supports inserting instructions for camera movement control. "I2V" for image to video. "T2V" for text to video. "MiniMax-Hailuo-02" is the latest model with best effect, ultra-clear quality and precise response.
        prompt (str): The prompt to generate the video from. When use Director model, the prompt supports 15 Camera Movement Instructions (Enumerated Values)
            -Truck: [Truck left], [Truck right]
            -Pan: [Pan left], [Pan right]
            -Push: [Push in], [Pull out]
            -Pedestal: [Pedestal up], [Pedestal down]
            -Tilt: [Tilt up], [Tilt down]
            -Zoom: [Zoom in], [Zoom out]
            -Shake: [Shake]
            -Follow: [Tracking shot]
            -Static: [Static shot]
        first_frame_image (str): The first frame image. The model must be "I2V" Series.
        duration (int, optional): The duration of the video. The model must be "MiniMax-Hailuo-02". Values can be 6 and 10.
        resolution (str, optional): The resolution of the video. The model must be "MiniMax-Hailuo-02". Values range ["768P", "1080P"]
        output_directory (str): The directory to save the video to.
        async_mode (bool, optional): Whether to use async mode. Defaults to False. If True, the video generation task will be submitted asynchronously and the response will return a task_id. Should use `query_video_generation` tool to check the status of the task and get the result.
    Returns:
        Text content with the path to the output video file.
    """
)
def generate_video(
    model: str = DEFAULT_T2V_MODEL,
    prompt: str = "",
    first_frame_image  = None,
    duration: int = None,
    resolution: str = None,
    output_directory: str = None,
    async_mode: bool = False
):
    try:
        if not prompt:
            raise MinimaxRequestError("Prompt is required")

        # check first_frame_image
        if first_frame_image:
            if not isinstance(first_frame_image, str):
                raise MinimaxRequestError(f"First frame image must be a string, got {type(first_frame_image)}")
            if not first_frame_image.startswith(("http://", "https://", "data:")):
                # if local image, convert to dataurl
                if not os.path.exists(first_frame_image):
                    raise MinimaxRequestError(f"First frame image does not exist: {first_frame_image}")
                with open(first_frame_image, "rb") as f:
                    image_data = f.read()
                    first_frame_image = f"data:image/jpeg;base64,{base64.b64encode(image_data).decode('utf-8')}"

        # step1: submit video generation task
        payload = {
            "model": model,
            "prompt": prompt
        }
        if first_frame_image:
            payload["first_frame_image"] = first_frame_image
        if duration:
            payload["duration"] = duration
        if resolution:
            payload["resolution"] = resolution
        response_data = api_client.post("/v1/video_generation", json=payload)
        task_id = response_data.get("task_id")
        if not task_id:
            raise MinimaxRequestError("Failed to get task_id from response")

        if async_mode:
            return TextContent(
                type="text",
                text=f"Success. Video generation task submitted: Task ID: {task_id}. Please use `query_video_generation` tool to check the status of the task and get the result."
            )

        # step2: wait for video generation task to complete
        file_id = None
        max_retries = 30  # 10 minutes total (30 * 20 seconds)
        retry_interval = 20  # seconds


        # MiniMax-Hailuo-02 model has a longer processing time, so we need to wait for a longer time
        if model == "MiniMax-Hailuo-02":
            max_retries = 60

        for attempt in range(max_retries):
            status_response = api_client.get(f"/v1/query/video_generation?task_id={task_id}")
            status = status_response.get("status")
            
            if status == "Fail":
                raise MinimaxRequestError(f"Video generation failed for task_id: {task_id}")
            elif status == "Success":
                file_id = status_response.get("file_id")
                if file_id:
                    break
                raise MinimaxRequestError(f"Missing file_id in success response for task_id: {task_id}")
            
            # Still processing, wait and retry
            time.sleep(retry_interval)

        if not file_id:
            raise MinimaxRequestError(f"Failed to get file_id for task_id: {task_id}")

        # step3: fetch video result
        file_response = api_client.get(f"/v1/files/retrieve?file_id={file_id}")
        download_url = file_response.get("file", {}).get("download_url")
        
        if not download_url:
            raise MinimaxRequestError(f"Failed to get download URL for file_id: {file_id}")
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Video URL: {download_url}"
            )
        # step4: download and save video
        output_path = build_output_path(output_directory, base_path)
        output_file_name = build_output_file("video", task_id, output_path, "mp4", True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_response = requests.get(download_url)
        video_response.raise_for_status()
        
        with open(output_path / output_file_name, "wb") as f:
            f.write(video_response.content)

        return TextContent(
            type="text",
            text=f"Success. Video saved as: {output_path / output_file_name}"
        )

    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to generate video: {str(e)}"
        )
    except (IOError, requests.RequestException) as e:
        return TextContent(
            type="text",
            text=f"Failed to handle video file: {str(e)}"
        )
    except Exception as e:
        return TextContent(
            type="text",
            text=f"Unexpected error while generating video: {str(e)}"
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Query Video Generation",
        readOnlyHint=True,
        openWorldHint=True,
    ),
    description="""Query the status of a video generation task.

    Args:
        task_id (str): The task ID to query. Should be the task_id returned by `generate_video` tool if `async_mode` is True.
        output_directory (str): The directory to save the video to.
    Returns:
        Text content with the status of the task.
    """
)
def query_video_generation(task_id: str, output_directory: str = None) -> TextContent:
    try:
        file_id = None
        response_data = api_client.get(f"/v1/query/video_generation?task_id={task_id}")
        status = response_data.get("status")
        if status == "Fail":
            return TextContent(
                type="text",
                text=f"Video generation FAILED for task_id: {task_id}"
            )
        elif status == "Success":
            file_id = response_data.get("file_id")
            if not file_id:
                raise MinimaxRequestError(f"Missing file_id in success response for task_id: {task_id}")
        else:
            return TextContent(
                type="text",
                text=f"Video generation task is still processing: Task ID: {task_id}"
            )
        file_response = api_client.get(f"/v1/files/retrieve?file_id={file_id}")
        download_url = file_response.get("file", {}).get("download_url")
        if not download_url:
            raise MinimaxRequestError(f"Failed to get download URL for file_id: {file_id}")
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Video URL: {download_url}"
            )
        output_path = build_output_path(output_directory, base_path)
        output_file_name = build_output_file("video", task_id, output_path, "mp4", True)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        video_response = requests.get(download_url)
        video_response.raise_for_status()

        with open(output_path / output_file_name, "wb") as f:
            f.write(video_response.content)

        return TextContent(
            type="text",
            text=f"Success. Video saved as: {output_path / output_file_name}"
        )
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to query video generation status: {str(e)}"
        )


@mcp.tool(
    annotations=ToolAnnotations(
        title="Text to Image",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Generate a image from a prompt.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

     Args:
        model (str, optional): The model to use. Values range ["image-01"], with "image-01" being the default.
        prompt (str): The prompt to generate the image from.
        aspect_ratio (str, optional): The aspect ratio of the image. Values range ["1:1", "16:9","4:3", "3:2", "2:3", "3:4", "9:16", "21:9"], with "1:1" being the default.
        n (int, optional): The number of images to generate. Values range [1, 9], with 1 being the default.
        prompt_optimizer (bool, optional): Whether to optimize the prompt. Values range [True, False], with True being the default.
        output_directory (str): The directory to save the image to.
    Returns:
        Text content with the path to the output image file.
    """
)
def text_to_image(
    model: str = DEFAULT_T2I_MODEL,
    prompt: str = "",
    aspect_ratio: str = "1:1",
    n: int = 1,
    prompt_optimizer: bool = True,
    output_directory: str = None,
):
    try:
        if not prompt:
            raise MinimaxRequestError("Prompt is required")

        payload = {
            "model": model, 
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "n": n,
            "prompt_optimizer": prompt_optimizer
        }

        response_data = api_client.post("/v1/image_generation", json=payload)
        image_urls = response_data.get("data",{}).get("image_urls",[])
        
        if not image_urls:
            raise MinimaxRequestError("No images generated")
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Image URLs: {image_urls}"
            )
        output_path = build_output_path(output_directory, base_path)
        output_file_names = []
        
        for i, image_url in enumerate(image_urls):
            output_file_name = build_output_file("image", f"{i}_{prompt}", output_path, "jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            
            with open(output_file_name, 'wb') as f:
                f.write(image_response.content)
            output_file_names.append(output_file_name)
            
        return TextContent(
            type="text",
            text=f"Success. Images saved as: {output_file_names}"
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to generate images: {str(e)}"
        )
    except (IOError, requests.RequestException) as e:
        return TextContent(
            type="text",
            text=f"Failed to save images: {str(e)}"
        )

@mcp.tool(
    annotations=ToolAnnotations(
        title="Music Generation",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Create a music generation task using AI models. Generate music from prompt and lyrics.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

    Args:
        prompt (str): Music creation inspiration describing style, mood, scene, etc.
            Example: "Pop music, sad, suitable for rainy nights". Character range: [10, 300]
        lyrics (str): Song lyrics for music generation.
            Use newline (\\n) to separate each line of lyrics. Supports lyric structure tags [Intro][Verse][Chorus][Bridge][Outro]
            to enhance musicality. Character range: [10, 600] (each Chinese character, punctuation, and letter counts as 1 character)
        stream (bool, optional): Whether to enable streaming mode. Defaults to False
        sample_rate (int, optional): Sample rate of generated music. Values: [16000, 24000, 32000, 44100]
        bitrate (int, optional): Bitrate of generated music. Values: [32000, 64000, 128000, 256000]
        format (str, optional): Format of generated music. Values: ["mp3", "wav", "pcm"]. Defaults to "mp3"
        output_directory (str, optional): Directory to save the generated music file

    Note: Currently supports generating music up to 1 minute in length.

    Returns:
        Text content with the path to the generated music file or generation status.
    """
)
def music_generation(
    prompt: str,
    lyrics: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bitrate: int = DEFAULT_BITRATE,
    format: str = DEFAULT_FORMAT,
    output_directory: str = None
) -> TextContent:
        try:
            # prompt and lyrics params check
            if not prompt:
                raise MinimaxRequestError("Prompt is required.")
            if not lyrics:
                raise MinimaxRequestError("Lyrics is required.")
            
            # Build request payload
            payload = {
                "model": DEFAULT_MUSIC_MODEL,
                "prompt": prompt,
                "lyrics": lyrics,
                "audio_setting": {
                    "sample_rate": sample_rate,
                    "bitrate": bitrate,
                    "format": format
                },
            }
            if resource_mode == RESOURCE_MODE_URL:
                payload["output_format"] = "url"

            # Call music generation API
            response_data = api_client.post("/v1/music_generation", json=payload)
                    
            # Handle response
            data = response_data.get('data', {})
            audio_hex = data.get('audio', '')

            if resource_mode == RESOURCE_MODE_URL:
                return TextContent(
                    type="text",
                    text=f"Success. Music url: {audio_hex}"
                )

            output_path = build_output_path(output_directory, base_path)
            output_file_name = build_output_file("music", f"{prompt}", output_path, format)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # hex->bytes
            audio_bytes = bytes.fromhex(audio_hex)

            # save audio to file
            with open(output_path / output_file_name, "wb") as f:
                f.write(audio_bytes)

            return TextContent(
                type="text",
                text=f"Success. Music saved as: {output_path / output_file_name}"
            )
        
        except MinimaxAPIError as e:
            return TextContent(
                type="text",
                text=f"Failed to generate music: {str(e)}"
            )
        except (IOError, requests.RequestException) as e:
            return TextContent(
                type="text",
                text=f"Failed to save music: {str(e)}"
        )

@mcp.tool(
    annotations=ToolAnnotations(
        title="Voice Design",
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=True,
    ),
    description="""Generate a voice based on description prompts.

    COST WARNING: This tool makes an API call to Minimax which may incur costs. Only use when explicitly requested by the user.

     Args:
        prompt (str): The prompt to generate the voice from.
        preview_text (str): The text to preview the voice.
        voice_id (str, optional): The id of the voice to use. For example, "male-qn-qingse"/"audiobook_female_1"/"cute_boy"/"Charming_Lady"...
        output_directory (str, optional): The directory to save the voice to.
    Returns:
        Text content with the path to the output voice file.
    """
)
def voice_design(
    prompt: str,
    preview_text: str,
    voice_id: str = None,
    output_directory: str = None,
):
    try:
        if not prompt:
            raise MinimaxRequestError("prompt is required")
        if not preview_text:
            raise MinimaxRequestError("preview_text is required")

       # Build request payload
        payload = {
            "prompt": prompt,
            "preview_text": preview_text
        }
        
        # Add voice_id if provided
        if voice_id:
            payload["voice_id"] = voice_id

        # Call voice design API
        response_data = api_client.post("/v1/voice_design", json=payload)

        # Get the response data
        generated_voice_id = response_data.get('voice_id', '')
        trial_audio_hex = response_data.get('trial_audio', '')
        
        if not generated_voice_id:
            raise MinimaxRequestError("No voice generated")
        if resource_mode == RESOURCE_MODE_URL:
            return TextContent(
                type="text",
                text=f"Success. Voice ID generated: {generated_voice_id}, Trial Audio: {trial_audio_hex}"
            )
        
        # hex->bytes
        audio_bytes = bytes.fromhex(trial_audio_hex)

        # save audio to file
        output_path = build_output_path(output_directory, base_path)
        output_file_name = build_output_file("voice_design", preview_text, output_path, "mp3")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path / output_file_name, "wb") as f:
            f.write(audio_bytes)

        return TextContent(
            type="text",
            text=f"Success. File saved as: {output_path / output_file_name}. Voice ID generated: {generated_voice_id}",
        )
        
    except MinimaxAPIError as e:
        return TextContent(
            type="text",
            text=f"Failed to design voice: {str(e)}"
        )

def main():
    print("Starting Minimax MCP server")
    """Run the Minimax MCP server"""
    mcp.run()


if __name__ == "__main__":
    main()
