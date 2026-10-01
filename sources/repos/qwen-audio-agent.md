# qwen-audio-agent

> 来源归档（ingest）

- **标题：** Qwen Audio Agent — Real-time Voice Runtime for AI Agents
- **类型：** repo + npm 包 + 桌面应用 + 技术报告
- **组织：** [QwenAudio](https://github.com/QwenAudio)（阿里巴巴通义音频 Agent 生态）
- **代码：** https://github.com/QwenAudio/qwen-audio-agent
- **论文：** https://arxiv.org/abs/2609.25195
- **文档 / 项目页：** https://qwenaudio.github.io/qwen-audio-agent/
- **Quickstart：** https://qwenaudio.github.io/qwen-audio-agent/getting-started/quickstart
- **入库日期：** 2026-10-01
- **一句话说明：** Node **≥22.22.2** 的 **全双工语音 Gateway**：可换 Realtime 前台（Qwen / OpenAI / Gemini / 本地 S2S 等），经 **Orchestration Runtime** 并行对话与后台 Agent 任务（ACP/A2A），提供 WebUI / TUI / 桌面悬浮球与示例（座舱、客服、数字人）。
- **沉淀到 wiki：** [Qwen-Audio-Agent](../../wiki/entities/paper-qwen-audio-agent.md)

---

## 开源状态（步骤 2.5，2026-10-01）

| 项 | 核查 |
|----|------|
| **GitHub** | 公开仓 [QwenAudio/qwen-audio-agent](https://github.com/QwenAudio/qwen-audio-agent)，MIT License，CI badge 绿 |
| **项目文档站** | [qwenaudio.github.io/qwen-audio-agent](https://qwenaudio.github.io/qwen-audio-agent/) 链到 GitHub、npm、Quickstart |
| **可运行入口** | CLI `qwenaudio`（Gateway）、`qwenaudio webui` / `tui`；`config.env` 配置 Realtime Provider 与 `AGENT_PROTOCOL` |
| **模型权重** | **不包含** 在仓库内；语音前台 / 后台 LLM 依赖各云 API Key 或本地 S2S 服务 URL |
| **结论** | **已开源**（编排运行时 + 客户端 + 示例）；模型与 API 凭证需自行配置 |

---

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [人形智能语音交互](../../wiki/methods/humanoid-voice-interaction.md) | 课程式 **ASR→LLM→技能→TTS** 闭环的 **工业级全双工 + 异步委派** 对照：打断、并行任务、结果回插 |
| [语音交互流水线](../../wiki/queries/humanoid-voice-interaction-pipeline.md) | Gateway 把「对话环」与「长任务环」解耦，可借鉴 barge-in 与进度查询状态机 |
| [VoiceStudio](../../wiki/entities/voicestudio.md) | 本地 TTS/ASR 工作台；Qwen-Audio-Agent 偏 **Realtime 端到端语音 Agent 运行时**，非单纯合成工具 |

---

## 仓库结构（Quickstart / README 归纳）

| 路径 / 组件 | 角色 |
|-------------|------|
| `server` / Gateway | 长驻 **Orchestration Runtime**：语音会话、任务状态、事件与记忆 |
| Realtime Provider 适配 | 对接 Qwen Audio Realtime、GPT-Live、Gemini Live、HF speech-to-speech 等 |
| Backend 适配 | ACP / A2A / Qwen Code 等；`AGENT_PROTOCOL` 选择 |
| `web` / `tui` / `desktop` | 客户端：浏览器、终端、macOS/Win/Linux 悬浮球 |
| `examples` | 智能座舱、客服、数字人等场景样例 |

**最短体验路径：** `qwenaudio config` → 编辑 `~/.config/qwaudio/config.env`（如 `DASHSCOPE_API_KEY` + `QWEN_AUDIO_REALTIME_MODEL`）→ `qwenaudio` → 另开终端 `qwenaudio webui` → 麦克风对话；加后台则配置 `AGENT_PROTOCOL=qwen` 等并重启 Gateway。

---

## 对 wiki 的映射

- 新建 **`wiki/entities/paper-qwen-audio-agent.md`**：技术报告 + 开源运行时实体页（架构 Mermaid、座舱 benchmark、源码时序图）。
- 轻量更新 **`wiki/methods/humanoid-voice-interaction.md`**：关联页增加本条目。

---

## 外部参考

```bibtex
@misc{qwenaudioagent2026,
  title={Qwen-Audio-Agent Technical Report},
  author={Chong Deng and Yunjie Ji and Yuxiang Kong and others},
  year={2026},
  eprint={2609.25195},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2609.25195},
}
```
