# OpenLess 官网

> 来源归档

- **标题：** OpenLess — 开源跨平台语音输入
- **类型：** site / product
- **链接：** <https://openless.top/>
- **代码：** <https://github.com/Open-Less/openless>
- **入库日期：** 2026-09-12
- **一句话说明：** 按住全局热键说话、松开后在任意应用光标处插入 AI 润色文本；macOS / Windows / Linux 本地优先，自带 ASR + LLM 凭证。
- **沉淀到 wiki：** [`wiki/entities/openless.md`](../../wiki/entities/openless.md)

## 开源状态（步骤 2.5）

- **代码：** [Open-Less/openless](https://github.com/Open-Less/openless) 公开；Release 提供 DMG / EXE / AppImage / deb / rpm；Homebrew Cask `brew install --cask openless`。
- **本地 ASR：** 捆绑 Qwen3-ASR（macOS）；Windows 实验性 Foundry Local Whisper / sherpa-onnx。
- **结论：** **已开源**（GitHub 元数据许可 AGPL-3.0；官网自述 MIT——以仓库 `LICENSE` 为准）。

## 对本库的意义

- 桌面端 **ASR → LLM 润色 → 光标注入** 闭环参考，与 [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md) 中 ASR 环节对照（本工具不做机器人技能路由）。
- 机器人研究与工程写作场景：在 Cursor / ChatGPT / Notion 等输入框口述结构化 Prompt、commit message、规格说明。
