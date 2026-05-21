# SenseNova-Skills（OpenSenseNova/SenseNova-Skills）

> 来源归档

- **标题：** SenseNova-Skills
- **类型：** repo
- **作者 / 组织：** OpenSenseNova（商汤 SenseNova 生态）
- **链接：** https://github.com/OpenSenseNova/SenseNova-Skills
- **安装指南：** https://github.com/OpenSenseNova/SenseNova-Skills/blob/main/INSTALL.md
- **入库日期：** 2026-05-21
- **一句话说明：** 遵循 [Agent Skills](https://agentskills.io/) 约定的模块化办公技能库：覆盖信息图/PPT/Excel 数据分析/深度研究与多源搜索，可与 OpenClaw、Hermes Agent 及 SenseNova 平台 API 组合成端到端办公工作流。
- **为什么值得保留：** 与本站 [Hermes Agent](../../wiki/entities/hermes-agent.md)、[mattpocock/skills](mattpocock-skills.md) 同属 **agentskills.io 技能片** 范式，但场景为 **办公生产力**（非纯编码）；含 **分层技能编排**（Tier 0 基础层 + Tier 1 业务技能 + 统一入口）、**可恢复的深度研究管线** 与 **数据分析→研究→PPT 全链示例**；`examples/embodied-ai-deep-research` 对具身/机器人行业调研有可直接复用的模板价值。
- **沉淀到 wiki：** 是 → [`wiki/entities/sensenova-skills.md`](../../wiki/entities/sensenova-skills.md)

## README 要点（归纳）

- **定位：** SenseNova 模型族可直接接入 [OpenClaw](https://openclaw.ai/)、[hermes-agent](https://github.com/NousResearch/hermes-agent) 等代理运行时；本仓每个技能独立目录，以 `SKILL.md` 声明触发条件、能力与执行流。
- **能力域：** 图像生成与可视化、幻灯片（PPT）、Excel 数据分析、深度研究（含学术/代码/中英文社交搜索子技能）。
- **商业封装：** 完整技能套件亦打包进 [**Raccoon**](https://office.xiaohuanxiong.com/home)（小浣熊办公），免自建环境与 API；本归档以 **开源仓库可自托管** 为主视角。
- **推荐组合：** Agent Skills 兼容运行时 + [SenseNova Platform API](https://platform.sensenova.cn/token-plan)（有免费 token 计划）。
- **安装路径：**
  - OpenClaw：`~/.openclaw/skills/`
  - Hermes：`~/.hermes/skills/`
  - 可让代理 clone 仓库并复制 `skills/*`，或手动 `git clone` + `cp -r SenseNova-Skills/skills/* <target>/`
- **技能分层（图像栈示例）：** `sn-image-base`（Tier 0：文生图/识别/文本优化，统一 `sn_agent_runner.py`）→ `sn-infographic`、`sn-image-imitate`、`sn-image-resume`（Tier 1）；各域有 `*-doctor` 环境自检技能。
- **PPT 管线：** `sn-ppt-entry` 统一入口（角色/受众/场景/页数/创意或标准模式）→ `sn-ppt-creative`（整页 PNG）或 `sn-ppt-standard`（style_spec → HTML 逐页 → VLM QC → PPTX）。
- **数据分析：** `sn-da-excel-workflow` 编排多表清洗/聚合；≥10k 行走 `sn-da-large-file-analysis`；图像输入走 `sn-da-image-caption`。
- **深度研究：** `sn-deep-research` 入口 → planning → 分维度证据 → synthesis → `report.md`；可恢复执行、产物持久化到 `report_dir`。
- **示例：** `examples/memory-price-end2end-analysis`（Excel → 深度研究 → PPT 全链）；`examples/embodied-ai-deep-research`（具身 AI 行业研究）；`examples/employee-performance-analysis`、`examples/property-fee-pricing-ppt` 等。
- **协议：** MIT（见仓库 `LICENSE`）。

## 对 wiki 的映射

| 主题 | 建议/wiki 页 |
|------|----------------|
| Agent Skills 办公技能栈 | [`wiki/entities/sensenova-skills.md`](../../wiki/entities/sensenova-skills.md) |
| 推荐运行时 | [`wiki/entities/hermes-agent.md`](../../wiki/entities/hermes-agent.md) |
| 编码向技能对照 | [`wiki/entities/mattpocock-skills.md`](../../wiki/entities/mattpocock-skills.md) |
| 外网读搜脚手架 | [`wiki/entities/agent-reach.md`](../../wiki/entities/agent-reach.md) |
| LLM 知识编译范式 | [`wiki/references/llm-wiki-karpathy.md`](../../wiki/references/llm-wiki-karpathy.md) |

## 与本站 sources 的其它锚点

- 运行时对照：[nousresearch_hermes_agent.md](nousresearch_hermes_agent.md)
- 编码技能对照：[mattpocock-skills.md](mattpocock-skills.md)
- 流程方法论对照：[obra-superpowers.md](obra-superpowers.md)
