# VoiceStudio 官方仓库

> 来源归档（repo）

- **标题：** VoiceStudio — Open-source, fully-local voice cloning, TTS, dubbing, dictation & transcription
- **类型：** repo / speech / tts / asr / voice-cloning / desktop / mcp / local-first
- **维护者：** [debpalash](https://github.com/debpalash)（独立开源项目）
- **代码：** <https://github.com/debpalash/VoiceStudio>（**已开源**，AGPL-3.0）
- **官网 / 安装：** <https://voicestudio.sh/> · [Releases](https://github.com/debpalash/VoiceStudio/releases/latest)
- **入库日期：** 2026-09-30
- **一句话说明：** 全本地 ElevenLabs 替代方案：Electron 桌面 + 多引擎 TTS/克隆/转写/配音，646 语言；可选 Local API 与 MCP 供 Agent 调用。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-09-30） |
|----|-------------------|
| **GitHub** | 公开仓 [debpalash/VoiceStudio](https://github.com/debpalash/VoiceStudio)；CI、Electron 安装脚本、Python API 后端均在仓内 |
| **权重 / 模型** | **按需下载**（各引擎独立许可）；README 要求商用前逐模型审阅 [LICENSE-NOTICE.md](https://github.com/debpalash/VoiceStudio/blob/main/LICENSE-NOTICE.md) |
| **桌面发行** | **已发布** GitHub Releases（Electron；0.5.3 后仅 Electron，Tauri 已退役） |
| **结论** | **已开源（应用 + 集成栈 + 文档）**；语音模型为第三方引擎与权重，非单一 bundled 权重 |

## 仓库入口（README 归纳）

| 组件 | 说明 |
|------|------|
| `electron/` | **唯一**桌面/Web UI 壳；`bun run dev` 开发预览 |
| `bun run setup:api` | 准备 Python 语音后端依赖 |
| `docs/speech-platform.md` | Local API（本地 HTTP） |
| `docs/mcp.md` | MCP Server，供 Claude Code / Cursor 等 Agent |
| `docs/engines/` | 多 TTS/ASR 引擎切换（默认 VoiceStudio / k2-fsa **OmniVoice**；转写默认 **WhisperX**） |
| `docs/install/agent.md` | 编码 Agent 一键安装与验收流程 |
| `curl … voicestudio.sh/install \| sh` | macOS / Linux 一键安装 Electron 发行版 |

## 技术要点（对 wiki 的映射）

1. **Local-first**：工作流默认跑在本机 GPU/CPU；远程 worker 与用量分析为可选且需同意。
2. **多引擎插件化**：TTS（CosyVoice、IndexTTS、MOSS-TTS、GPT-SoVITS、sherpa-onnx 等）与 ASR（Whisper 系、FunASR、Parakeet 等）可切换，见 `docs/feature-catalog.md`。
3. **产品能力**：语音克隆、Voice Design、视频配音、悬浮听写、说话人分离、批量队列、水印等。
4. **Agent 集成**：Local API + MCP + `npx skills add debpalash/VoiceStudio`（voicestudio / voicestudio-maintainer skills）。
5. **机器人关联**：可作 [人形语音交互](../../wiki/methods/humanoid-voice-interaction.md) 闭环中的 **本地 TTS/ASR 工作台** 或 **开发机侧语音 I/O**；真机侧仍需 ROS 2 / SDK 动作接口与 barge-in 状态机，见 [语音交互流水线](../../wiki/queries/humanoid-voice-interaction-pipeline.md)。

## 对 wiki 的映射

- 主实体：[VoiceStudio](../../wiki/entities/voicestudio.md)
- 站点归档：[voicestudio.md](../sites/voicestudio.md)
