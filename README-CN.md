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

<p align="center" style="line-height: 1.5; font-size: 18px; margin: 4px auto; text-decoration: underline;"><a href="README.md">English Version</a></p>

<p align="center">
  MiniMax官方模型上下文协议(MCP)服务器，支持与强大的文本转语音和视频/图像生成API交互。允许MCP客户端如<a href="https://www.anthropic.com/claude">Claude Desktop</a>、<a href="https://www.cursor.so">Cursor</a>、<a href="https://codeium.com/windsurf">Windsurf</a>、<a href="https://github.com/openai/openai-agents-python">OpenAI Agents</a>等生成语音、克隆声音、生成视频、生成图像等功能。
</p>

> 💡 推荐优先使用 **[MiniMax CLI (mmx-cli)](https://github.com/MiniMax-AI/cli)** — MiniMax 官方命令行工具，支持最新模型及更多功能（文本对话、图片理解、搜索、音乐翻唱等），支持作为 Claude Code、Cursor、OpenClaw 等 AI Agent 的技能接入。

## Documentation
- [English Documentation](README.md)
- [MiniMax-MCP-JS](https://github.com/MiniMax-AI/MiniMax-MCP-JS) - MiniMax MCP的官方JavaScript版本

## 快速开始使用 MCP 客户端
1. 从[MiniMax国内开放平台](https://platform.minimaxi.com/user-center/basic-information/interface-key)｜[MiniMax国际开放平台](https://www.minimax.io/platform/user-center/basic-information/interface-key)获取你的 API 密钥。
2. 安装`uv`（Python包管理器），使用`curl -LsSf https://astral.sh/uv/install.sh | sh`安装或查看`uv` [仓库](https://github.com/astral-sh/uv)获取其他安装方法。
3. **重要提示: API的服务器地址和密钥在不同区域有所不同**，两者需要匹配，否则会有 `invalid api key` 的错误

|地区| 国际  | 国内  |
|:--|:-----|:-----|
|MINIMAX_API_KEY| 获取密钥 [MiniMax国际版](https://www.minimax.io/platform/user-center/basic-information/interface-key) | 获取密钥 [MiniMax](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
|MINIMAX_API_HOST| https://api.minimax.io | https://api.minimaxi.com |


### Claude Desktop
前往`Claude > Settings > Developer > Edit Config > claude_desktop_config.json`包含以下内容：

```
{
  "mcpServers": {
    "MiniMax": {
      "command": "uvx",
      "args": [
        "minimax-mcp"
      ],
      "env": {
        "MINIMAX_API_KEY": "填写你的API密钥",
        "MINIMAX_MCP_BASE_PATH": "本地输出目录路径，如/User/xxx/Desktop",
        "MINIMAX_API_HOST": "填写API Host, https://api.minimaxi.com 或 https://api.minimax.io",
        "MINIMAX_API_RESOURCE_MODE": "可选配置，资源生成后的提供方式, 可选项为 [url|local], 默认为 url"
      }
    }
  }
}
```


⚠️ 注意：API Key需要与Host匹配。如果出现"API Error: invalid api key"错误，请检查您的API Host：
- 国际版Host：`https://api.minimax.io`
- 国内版Host：`https://api.minimaxi.com` 

如果你使用Windows，你需要在Claude Desktop中启用"开发者模式"才能使用MCP服务器。点击左上角汉堡菜单中的"Help"，然后选择"Enable Developer Mode"。


### Cursor
前往`Cursor -> Preferences -> Cursor Settings -> MCP -> Add new global MCP Server`添加上述配置。

你的MCP客户端现在可以通过Claude Desktop和Cursor等这些工具与MiniMax交互：

## Transport
我们支持两种传输方式: stdio and sse.
| stdio  | SSE  |
|:-----|:-----|
| 在本地部署运行 | 本地或云端部署均可  |
|通过 stdout 进行通信| 通过网络通信|
|输入：支持处理本地文件，或有效的URL资源| 输入: 若部署在云端，建议使用URL进行输入|

## 可用方法
| 方法  | 描述  |
|-|-|
|`text_to_audio`|使用指定音色将文本生成音频|
|`list_voices`|查询所有可用音色|
|`voice_clone`|根据指定音频文件克隆音色|
|`generate_video`|根据指定 prompt 生成视频|
|`text_to_image`|根据指定 prompt 生成图片|
|`music_generation`|根据指定 prompt 和歌词生成音乐|
|`voice_design`|根据指定 prompt 生成音色和试听文本|

## 更新日志

### 2025年7月2日

#### 🆕 新增功能
- **音色设计**: 新增 `voice_design` 工具 - 根据描述性提示词创建自定义音色并生成试听音频
- **视频生成增强**: 新增 `MiniMax-Hailuo-02` 模型，支持超清画质和时长/分辨率控制
- **音乐生成**: 采用 `music-1.5` 模型增强 `music_generation` 工具

#### 📈 功能增强
- `voice_design` - 根据文本描述生成个性化音色
- `generate_video` - 现在支持 MiniMax-Hailuo-02 模型，可选择 6s/10s 时长和 768P/1080P 分辨率
- `music_generation` - 采用 music-1.5 模型进行高质量音乐创作

## FAQ
### 1. invalid api key
请检查你获取的 API Key 和填写的 API Host 是否是同一地区的：
|地区| 国际  | 国内  |
|:--|:-----|:-----|
|MINIMAX_API_KEY| 获取密钥 [MiniMax国际版](https://www.minimax.io/platform/user-center/basic-information/interface-key) | 获取密钥 [MiniMax](https://platform.minimaxi.com/user-center/basic-information/interface-key) |
|MINIMAX_API_HOST| https://api.minimax.io | https://api.minimaxi.com

### 2. spawn uvx ENOENT
请在你的终端输入一下命令，查看uvx命令的绝对路径：
```sh
which uvx
```
如果得到如下的输出 (如：/usr/local/bin/uvx)，更新mcp配置 ("command": "/usr/local/bin/uvx"). 

### 3. 如何用 `generate_video` 工具异步生成视频
在对话前设置一些规则:
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/cursor_rule2.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
或者放到本地客户端的规则中 (以 Cursor 为例):
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/cursor_video_rule.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>


## 使用示例

⚠️ 注意：使用这些工具可能会产生费用。

### 1. 播报晚间新闻片段
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_20-07-53.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>

### 2. 克隆声音
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-45-13.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>

### 3. 生成视频
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-58-52.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/Snipaste_2025-04-09_19-59-43.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle; "/>

### 4. 生成图像
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/gen_image.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle;"/>
<img src="https://public-cdn-video-data-algeng.oss-cn-wulanchabu.aliyuncs.com/gen_image1.png?x-oss-process=image/resize,p_50/format,webp" style="display: inline-block; vertical-align: middle; "/>
