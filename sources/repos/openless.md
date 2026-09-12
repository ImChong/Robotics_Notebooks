# Open-Less/openless

> 来源归档

- **标题：** OpenLess
- **类型：** repo
- **组织：** Open-Less
- **链接：** <https://github.com/Open-Less/openless>
- **官网：** <https://openless.top/>
- **许可：** AGPL-3.0（GitHub API；以仓库 `LICENSE` 为准）
- **入库日期：** 2026-09-12
- **一句话说明：** Tauri 2（Rust + React/TS）跨平台语音输入：全局热键录音 → 云/本地 ASR → LLM 润色（含 AI-Prompt 结构化模式）→ 流式插入光标；Style Pack 市场与可学习词典。
- **沉淀到 wiki：** [`wiki/entities/openless.md`](../../wiki/entities/openless.md)

## 开源状态

**已开源**：完整应用源码在 `openless-all/app/`；macOS 本地 ASR 依赖 vendored [`Open-Less/qwen-asr`](https://github.com/Open-Less/qwen-asr) 子模块。

## 技术要点（README 摘要）

| 维度 | 内容 |
|------|------|
| 栈 | Tauri 2 + Rust 后端 + React/TypeScript 前端 |
| 平台 | macOS 14+、Windows 10+、Linux（AppImage/deb/rpm）、Android APK |
| 云 ASR | 火山引擎、讯飞 RTASR、阿里百炼、阶跃 StepAudio、智谱 GLM-ASR、小米 MiMo、ElevenLabs Scribe、OpenAI 兼容（Whisper/Groq/SenseVoice 等）、Apple Speech |
| 润色 | Ark、DeepSeek、OpenAI、Gemini、SiliconFlow、OpenRouter、MiniMax、阶跃等 + 任意 OpenAI 兼容端点 |
| 输出模式 | 原文 / 轻润色 / 结构化（AI-Prompt）/ 正式；独立翻译热键 |
| 凭证 | OS 密钥库（Keychain / Credential Manager / Linux keyring），`~/.openless` 仅存配置 |
| 分发 | GitHub Releases、Homebrew Cask、Windows installer、应用内 Tauri updater |

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [openless](../../wiki/entities/openless.md) | 实体页 |
| [humanoid-voice-interaction](../../wiki/methods/humanoid-voice-interaction.md) | 机器人端 ASR→技能闭环（对照本桌面工具） |
| [humanoid-voice-interaction-pipeline](../../wiki/queries/humanoid-voice-interaction-pipeline.md) | 人形语音工程流水线 |
| [openclaw](../../wiki/entities/openclaw.md) | 个人 AI 助手控制平面（语音→技能；与本工具「语音→润色文本」定位不同） |
| [agent-reach](../../wiki/entities/agent-reach.md) | 编码代理外网读搜脚手架（互补：本工具偏桌面口述写作） |

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/openless.md`](../../wiki/entities/openless.md)**；安装与构建命令以克隆时 README 为准。
