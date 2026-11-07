# speech model default values
DEFAULT_VOICE_ID = "female-shaonv"
DEFAULT_SPEECH_MODEL = "speech-2.8-hd"
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

# network timeouts (seconds); read timeout applies between received bytes,
# so large downloads are not aborted while they keep making progress
REQUEST_TIMEOUT = 120
