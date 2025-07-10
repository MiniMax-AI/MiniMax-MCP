![export](https://github.com/MiniMax-AI/MiniMax-01/raw/main/figures/MiniMaxLogo-Light.png)

<div align="center" style="line-height: 1;">
  <a href="https://www.minimax.io" target="_blank" style="margin: 2px; color: var(--fgColor-default);">
    <img alt="Homepage" src="https://img.shields.io/badge/_Homepage-MiniMax-FF4040?style=flat-square&labelColor=2C3E50&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNDkwLjE2IDQxMS43Ij48ZGVmcz48c3R5bGU+LmNscy0xe2ZpbGw6I2ZmZjt9PC9zdHlsZT48L2RlZnM+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjMzLjQ1LDQwLjgxYTE3LjU1LDE3LjU1LDAsMSwwLTM1LjEsMFYzMzEuNTZhNDAuODIsNDAuODIsMCwwLDEtODEuNjMsMFYxNDVhMTcuNTUsMTcuNTUsMCwxLDAtMzUuMDksMHY3OS4wNmE0MC44Miw0MC44MiwwLDAsMS04MS42MywwVjE5NS40MmExMS42MywxMS42MywwLDAsMSwyMy4yNiwwdjI4LjY2YTE3LjU1LDE3LjU1LDAsMCwwLDM1LjEsMFYxNDVBNDAuODIsNDAuODIsMCwwLDEsMTQwLDE0NVYzMzEuNTZhMTcuNTUsMTcuNTUsMCwwLDAsMzUuMSwwVjIxNy41aDBWNDAuODFhNDAuODEsNDAuODEsMCwxLDEsODEuNjIsMFYyODEuNTZhMTEuNjMsMTEuNjMsMCwxLDEtMjMuMjYsMFptMjE1LjksNjMuNEE0MC44Niw0MC44NiwwLDAsMCw0MDguNTMsMTQ1VjMwMC44NWExNy41NSwxNy41NSwwLDAsMS0zNS4wOSwwdi0yNjBhNDAuODIsNDAuODIsMCwwLDAtODEuNjMsMFYzNzAuODlhMTcuNTUsMTcuNTUsMCwwLDEtMzUuMSwwVjMzMGExMS42MywxMS42MywwLDEsMC0yMy4yNiwwdjQwLjg2YTQwLjgxLDQwLjgxLDAsMCwwLDgxLjYyLDBWNDAuODFhMTcuNTUsMTcuNTUsMCwwLDEsMzUuMSwwdjI2MGE0MC44Miw0MC44MiwwLDAsMCw4MS42MywwVjE0NWExNy41NSwxNy41NSwwLDEsMSwzNS4xLDBWMjgxLjU2YTExLjYzLDExLjYzLDAsMCwwLDIzLjI2LDBWMTQ1QTQwLjg1LDQwLjg1LDAsMCwwLDQ0OS4zNSwxMDQuMjFaIi8+PC9zdmc+&logoWidth=20" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://arxiv.org/abs/2501.08313" target="_blank" style="margin: 2px;">
    <img alt="Paper" src="https://img.shields.io/badge/📖_Paper-MiniMax--01-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
   <a href="https://chat.minimax.io/" target="_blank" style="margin: 2px;">
    <img alt="Chat" src="https://img.shields.io/badge/_MiniMax_Chat-FF4040?style=flat-square&labelColor=2C3E50&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHhtbG5zOnhsaW5rPSJodHRwOi8vd3d3LnczLm9yZy8xOTk5L3hsaW5rIiB2aWV3Qm94PSIwIDAgNDkwLjE2IDQxMS43Ij48ZGVmcz48c3R5bGU+LmNscy0xe2ZpbGw6I2ZmZjt9PC9zdHlsZT48L2RlZnM+PHBhdGggY2xhc3M9ImNscy0xIiBkPSJNMjMzLjQ1LDQwLjgxYTE3LjU1LDE3LjU1LDAsMSwwLTM1LjEsMFYzMzEuNTZhNDAuODIsNDAuODIsMCwwLDEtODEuNjMsMFYxNDVhMTcuNTUsMTcuNTUsMCwxLDAtMzUuMDksMHY3OS4wNmE0MC44Miw0MC44MiwwLDAsMS04MS42MywwVjE5NS40MmExMS42MywxMS42MywwLDAsMSwyMy4yNiwwdjI4LjY2YTE3LjU1LDE3LjU1LDAsMCwwLDM1LjEsMFYxNDVBNDAuODIsNDAuODIsMCwwLDEsMTQwLDE0NVYzMzEuNTZhMTcuNTUsMTcuNTUsMCwwLDAsMzUuMSwwVjIxNy41aDBWNDAuODFhNDAuODEsNDAuODEsMCwxLDEsODEuNjIsMFYyODEuNTZhMTEuNjMsMTEuNjMsMCwxLDEtMjMuMjYsMFptMjE1LjksNjMuNEE0MC44Niw0MC44NiwwLDAsMCw0MDguNTMsMTQ1VjMwMC44NWExNy41NSwxNy41NSwwLDAsMS0zNS4wOSwwdi0yNjBhNDAuODIsNDAuODIsMCwwLDAtODEuNjMsMFYzNzAuODlhMTcuNTUsMTcuNTUsMCwwLDEtMzUuMSwwVjMzMGExMS42MywxMS42MywwLDEsMC0yMy4yNiwwdjQwLjg2YTQwLjgxLDQwLjgxLDAsMCwwLDgxLjYyLDBWNDAuODFhMTcuNTUsMTcuNTUsMCwwLDEsMzUuMSwwdjI2MGE0MC44Miw0MC44MiwwLDAsMCw4MS42MywwVjE0NWExNy41NSwxNy41NSwwLDEsMSwzNS4xLDBWMjgxLjU2YTExLjYzLDExLjYzLDAsMCwwLDIzLjI2LDBWMTQ1QTQwLjg1LDQwLjg1LDAsMCwwLDQ0OS4zNSwxMDQuMjFaIi8+PC9zdmc+&logoWidth=20" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.minimax.io/platform" style="margin: 2px;">
    <img alt="API" src="https://img.shields.io/badge/⚡_API-Platform-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>  
</div>
<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/MiniMaxAI" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face" src="https://img.shields.io/badge/🤗_Hugging_Face-MiniMax-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://github.com/MiniMax-AI/MiniMax-AI.github.io/blob/main/images/wechat-qrcode.jpeg" target="_blank" style="margin: 2px;">
    <img alt="WeChat" src="https://img.shields.io/badge/_WeChat-MiniMax-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
  <a href="https://www.modelscope.cn/organization/MiniMax" target="_blank" style="margin: 2px;">
    <img alt="ModelScope" src="https://img.shields.io/badge/_ModelScope-MiniMax-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>
<div align="center" style="line-height: 1;">
   <a href="https://github.com/MiniMax-AI/MiniMax-MCP/blob/main/LICENSE" style="margin: 2px;">
    <img alt="Code License" src="https://img.shields.io/badge/_Code_License-MIT-FF4040?style=flat-square&labelColor=2C3E50" style="display: inline-block; vertical-align: middle;"/>
  </a>
</div>

<p align="center">
  Official MiniMax Model Context Protocol (MCP) server that enables interaction with powerful Text to Speech and video/image generation APIs. This server allows MCP clients like <a href="https://www.anthropic.com/claude">Claude Desktop</a>, <a href="https://www.cursor.so">Cursor</a>, <a href="https://codeium.com/windsurf">Windsurf</a>, <a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a> and others to generate speech, clone voices, generate video, generate image and more.
</p>

## Documentation
- [中文文档](README-CN.md)
- [MiniMax-MCP-JS](https://github.com/MiniMax-AI/MiniMax-MCP-JS) - Official JavaScript implementation of MiniMax MCP

## Quickstart with MCP Client
1. Get your API key from [MiniMax](https://www.minimax.io/platform/user-center/basic-information/interface-key). 
2. Install `uv` (Python package manager), install with `curl -LsSf https://astral.sh/uv/install.sh | sh` or see the `uv` [repo](https://github.com/astral-sh/uv) for additional install methods.
3. **Important**: The API host and key vary by region and must match; otherwise, you'll encounter an `Invalid API key` error.

|Region| Global  | Mainland  |
|:--|:-----|:-----|
|MINIMAX_API_KEY| go get from [MiniMax Global](https://www.minimax.io/platform/user-center/basic-information/interface-key) | go get from [MiniMax](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
|MINIMAX_API_HOST| https://api.minimax.io | https://api.minimaxi.com |

### Installing via Smithery

To install MiniMax MCP Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/@MiniMax-AI/MiniMax-MCP):

```bash
npx -y @smithery/cli install @MiniMax-AI/MiniMax-MCP --client claude
```


### Claude Desktop
Go to `Claude > Settings > Developer > Edit Config > claude_desktop_config.json` to include the following:

```
{
  "mcpServers": {
    "MiniMax": {
      "command": "uvx",
      "args": [
        "minimax-mcp",
        "-y"
      ],
      "env": {
        "MINIMAX_API_KEY": "insert-your-api-key-here",
        "MINIMAX_MCP_BASE_PATH": "local-output-dir-path, such as /User/xxx/Desktop",
        "MINIMAX_API_HOST": "api host, https://api.minimax.io | https://api.minimaxi.com",
        "MINIMAX_API_RESOURCE_MODE": "optional, [url|local], url is default, audio/image/video are downloaded locally or provided in URL format"
      }
    }
  }
}

```
⚠️ Warning: The API key needs to match the host. If an error "API Error: invalid api key" occurs, please check your api host:
- Global Host：`https://api.minimax.io`
- Mainland Host：`https://api.minimaxi.com`

If you're using Windows, you will have to enable "Developer Mode" in Claude Desktop to use the MCP server. Click "Help" in the hamburger menu in the top left and select "Enable Developer Mode".


### Cursor
Go to `Cursor -> Preferences -> Cursor Settings -> MCP -> Add new global MCP Server` to add above config.

That's it. Your MCP client can now interact with MiniMax through these tools:

## Transport
We support two transport types: stdio and sse.
| stdio  | SSE  |
|:-----|:-----|
| Run locally | Can be deployed locally or in the cloud |
| Communication through `stdout` | Communication through `network` |
| Input: Supports processing `local files` or valid `URL` resources | Input: When deployed in the cloud, it is recommended to use `URL` for input |

## Available Tools
| tool  | description  |
|-|-|
|`text_to_audio`|Convert text to audio with a given voice|
|`list_voices`|List all voices available|
|`voice_clone`|Clone a voice using provided audio files|
|`generate_video`|Generate a video from a prompt|
|`text_to_image`|Generate a image from a prompt|
|`query_video_generation`|Query the result of video generation task|
|`music_generation`|Generate a music track from a prompt and lyrics|
|`voice_design`|Generate a voice from a prompt using preview text|

## Release Notes

### July 2, 2025

#### 🆕 What's New
- **Voice Design**: New `voice_design` tool - create custom voices from descriptive prompts with preview audio
- **Video Enhancement**: Added `MiniMax-Hailuo-02` model with ultra-clear quality and duration/resolution controls  
- **Music Generation**: Enhanced `music_generation` tool powered by `music-1.5` model

#### 📈 Enhanced Tools
- `voice_design` - Generate personalized voices from text descriptions
- `generate_video` - Now supports MiniMax-Hailuo-02 with 6s/10s duration and 768P/1080P resolution options
- `music_generation` - High-quality music creation with music-1.5 model

## FAQ
### 1. invalid api key
Please ensure your API key and API host are regionally aligned
|Region| Global  | Mainland  |
|:--|:-----|:-----|
|MINIMAX_API_KEY| go get from [MiniMax Global](https://www.minimax.io/platform/user-center/basic-information/interface-key) | go get from [MiniMax](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
|MINIMAX_API_HOST| https://api.minimax.io | https://api.minimaxi.com |

### 2. spawn uvx ENOENT
Please confirm its absolute path by running this command in your terminal:
```sh
which uvx
```
Once you obtain the absolute path (e.g., /usr/local/bin/uvx), update your configuration to use that path (e.g., "command": "/usr/local/bin/uvx"). 

### 3. How to use `generate_video` in async-mode
Define completion rules before starting:
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/cursor_rule2.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
Alternatively, these rules can be configured in your IDE settings (e.g., Cursor):
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/cursor_video_rule.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>


## Example usage

⚠️ Warning: Using these tools may incur costs.

### 1. broadcast a segment of the evening news
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_20-07-53.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>

### 2. clone a voice
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-45-13.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>

### 3. generate a video
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-58-52.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-59-43.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle; "/>

### 4. generate images
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/gen_image.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/gen_image1.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle; "/>
