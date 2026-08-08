# CLI-Anything: Towards Agent-Native Computer Use（arXiv:2606.03854）

> 来源归档（ingest）

- **标题：** CLI-Anything: Towards Agent-Native Computer Use
- **缩写 / 框架：** **CLI-Anything**；**CLI-Hub**；agent-native computer use
- **类型：** paper / tech-report / cs.HC / llm-agents / computer-use
- **arXiv：** <https://arxiv.org/abs/2606.03854>（v1 2026-06-02；PDF：<https://arxiv.org/pdf/2606.03854>）
- **项目页：** <https://hkuds.github.io/CLI-Anything/>
- **代码：** <https://github.com/HKUDS/CLI-Anything>（**已开源**，Apache-2.0）
- **作者：** Yuhao Yang、Tianyu Fan、Chao Huang
- **机构：** 香港大学（HKU）/ HKUDS 开源组织语境（以作者与 GitHub org 为准）
- **入库日期：** 2026-08-08
- **一句话说明：** 技术报告主张 **用 CLI harness 做 agent-native computer use**，批评 GUI 像素操控范式；并以 CLI-Hub 平台承载方法论、架构与生态基础设施。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-08-08） |
|----|-------------------|
| **项目页** | [CLI-Hub](https://hkuds.github.io/CLI-Anything/) 列代码仓与安装路径 |
| **代码** | [HKUDS/CLI-Anything](https://github.com/HKUDS/CLI-Anything) 公开；`CITATION.cff` 指向本 arXiv |
| **结论** | **已开源**（报告与平台一致落地）。本库以 **工具实体页** 升格为主，不另建完整 `paper-*` 深读页（深度论文拆解交给专题 Notebooks 习惯；此处保留主张与映射）。 |

## 摘录 1：问题与主张（Abstract）

- **痛点：** 主流 computer-use 走 GUI agent（截图 → 定位 UI → 鼠标点击），与模型擅长的结构化/程序化控制错位；像素级交互脆弱、依赖时序与坐标，界面一变即崩。
- **主张：** 不应强迫 agent 模拟人类感知极限，而应把现有应用改造成 **命令行 harness**：结构化命令、显式状态、确定性反馈，消除有损的「视觉→计算」翻译。
- **平台：** 报告同时介绍 **CLI-Hub** 作为将该愿景工程化的综合平台（方法论 + 架构 + 基础设施）。

**对 wiki 的映射：** 写入 [`wiki/entities/cli-anything.md`](../../wiki/entities/cli-anything.md)「为什么重要 / 核心原理」；与 GUI agent、MCP、Skills 对照表。

## 摘录 2：设计取向（与仓库实现对齐）

| 轴 | 要点 |
|----|------|
| **交互面** | 优先 CLI / JSON / REPL，而非截图点击 |
| **保真** | 调用真实软件后端（渲染、导出、编辑），避免「玩具重写丢 90% 能力」 |
| **可发现** | 每 harness 生成 `SKILL.md`，供 SKILL 兼容代理发现 |
| **可分发** | Hub 注册表 + `cli-hub` 包管理；生成器 7 阶段含测试与打包 |

**对 wiki 的映射：** 实体页「流程总览」画 Generate → Hub → Agent 调用闭环。

## 摘录 3：与本库读者的边界

- 本文是 **通用 computer-use / HCI** 技术报告，不是机器人控制或 RL 论文。
- 对本库价值在于：**CAD/3D/引擎/文档工具** 的 agent 化入口（FreeCAD、Blender、Godot 等 harness），以及与 [OpenClaw](../../wiki/entities/openclaw.md) / [Hermes](../../wiki/entities/hermes-agent.md) 技能生态的衔接。
- 勿把 CLI-Anything 写成「可替代真机安全闸门或 Robot Gateway」——它解决的是 **软件操控面**，不是物理执行契约。

## 建议 wiki 动作

1. 升格 [`wiki/entities/cli-anything.md`](../../wiki/entities/cli-anything.md)（工具实体，非 `paper-*`）
2. 互链仓库/站点归档与 Hermes、OpenClaw、Agent Reach、FreeCAD MCP
3. 不强制新建 `wiki/entities/paper-cli-anything.md`（避免与工具实体重复）；若后续需要对照 GUI-agent 基准数字再补论文页
