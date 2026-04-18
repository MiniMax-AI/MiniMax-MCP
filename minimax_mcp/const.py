# speech model default values
DEFAULT_VOICE_ID = "female-shaonv"
DEFAULT_SPEECH_MODEL = "speech-2.6-hd"
DEFAULT_MUSIC_MODEL = "music-2.0"
DEFAULT_SPEED = 1.0
DEFAULT_VOLUME = 1.0
DEFAULT_PITCH = 0
DEFAULT_EMOTION = "happy"
DEFAULT_SAMPLE_RATE = 32000
DEFAULT_BITRATE = 128000
DEFAULT_CHANNEL = 1
DEFAULT_FORMAT = "mp3"
DEFAULT_LANGUAGE_BOOST = "auto"

# video model default values
DEFAULT_T2V_MODEL = "MiniMax-Hailuo-2.3"

# image model default values
DEFAULT_T2I_MODEL = "image-01"

# ENV variables
ENV_MINIMAX_API_KEY = "MINIMAX_API_KEY"
ENV_MINIMAX_API_HOST = "MINIMAX_API_HOST"
ENV_MINIMAX_MCP_BASE_PATH = "MINIMAX_MCP_BASE_PATH"
ENV_RESOURCE_MODE = "MINIMAX_API_RESOURCE_MODE"

RESOURCE_MODE_LOCAL = "local" # save resource to local file system
RESOURCE_MODE_URL = "url" # provide resource url

ENV_FASTMCP_LOG_LEVEL = "FASTMCP_LOG_LEVEL"

# HTTP timeout (seconds)
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 120

# Valid parameter ranges
VALID_SAMPLE_RATES = {8000, 16000, 22050, 24000, 32000, 44100}
VALID_BITRATES = {32000, 64000, 128000, 256000}
VALID_EMOTIONS = {"happy", "sad", "angry", "fearful", "disgusted", "surprised", "neutral"}
VALID_AUDIO_FORMATS = {"pcm", "mp3", "flac"}
VALID_CHANNELS = {1, 2}
VALID_ASPECT_RATIOS = {"1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"}