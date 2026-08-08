# CLI-Anything（HKUDS/CLI-Anything）

> 来源归档（ingest）

- **标题：** CLI-Anything: Making ALL Software Agent-Native
- **类型：** repo / agent-infrastructure / cli / skills / computer-use
- **作者 / 组织：** [HKUDS](https://github.com/HKUDS)（香港大学数据智能相关开源组织）；技术报告作者 Yuhao Yang、Tianyu Fan、Chao Huang
- **代码：** <https://github.com/HKUDS/CLI-Anything>（**已开源**，Apache-2.0）
- **项目页 / CLI-Hub：** <https://hkuds.github.io/CLI-Anything/>（亦可 <https://clianything.cc/>）
- **技术报告：** <https://arxiv.org/abs/2606.03854>（*CLI-Anything: Towards Agent-Native Computer Use*）
- **PyPI：** `cli-anything-hub`（`pip install cli-anything-hub` → `cli-hub …`）
- **许可：** Apache-2.0（以仓库 `LICENSE` 为准）
- **入库日期：** 2026-08-08
- **一句话说明：** 用 **7 阶段 harness 生成管线** 把任意软件/代码库/API 变成 **agent-native CLI**（Click + JSON/人类双输出 + REPL + 测试 + `SKILL.md`），并以 **CLI-Hub** 注册表统一浏览安装；主张用结构化命令替代脆弱的 GUI 像素操控。

## 开源状态（步骤 2.5）

| 项 | 核查（2026-08-08） |
|----|-------------------|
| **GitHub** | 公开仓 [HKUDS/CLI-Anything](https://github.com/HKUDS/CLI-Anything)；主语言 Python；Apache-2.0 |
| **项目页** | [CLI-Hub](https://hkuds.github.io/CLI-Anything/) 可浏览/安装社区与官方 harness；演示含 FreeCAD / Blender / Draw.io 等 |
| **Tech Report** | arXiv:2606.03854；`CITATION.cff` 指向该文 |
| **结论** | **已开源**（生成器插件、大量应用 harness、Hub 包与 registry）。上游软件本身（Blender/GIMP/FreeCAD…）仍须本机另行安装。 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [CLI-Anything 实体页](../../wiki/entities/cli-anything.md) | 升格：定位、7 阶段管线、Hub、与 GUI agent / MCP / Skills 边界 |
| [Hermes Agent](../../wiki/entities/hermes-agent.md) | 宿主侧：仓内有 `hermes-skill`；Hermes 是常驻 agent OS，CLI-Anything 生成其可调用的软件 CLI |
| [OpenClaw](../../wiki/entities/openclaw.md) | SKILL 兼容宿主之一；Hub meta-skill 可经 `npx skills` 装入 |
| [Agent Reach](../../wiki/entities/agent-reach.md) | 互补：Reach 聚合**外网读搜** CLI；CLI-Anything 为**任意专业软件**生成可控 CLI |
| [FreeCAD MCP](../../wiki/entities/freecad-mcp.md) | 同域对照：MCP+RPC 桥接桌面 FreeCAD vs `cli-anything-freecad` 生成式 CLI harness |
| [Unreal MCP](../../wiki/entities/unreal-mcp.md) | 引擎侧另一类代理桥；Hub 亦收录 UE 相关社区 CLI |
| [HarnessBank](../../wiki/entities/paper-harnessbank.md) | 概念相邻：二者都谈 agent **harness**，但 HarnessBank 进化宿主表面，CLI-Anything 生成**应用侧** CLI harness |

## README / 架构要点（归纳，2026-08-08）

- **问题设定（Agent–Software Gap）：** GUI agent 依赖截图/坐标点击，脆弱且强迫模型模拟人类感知；CLI-Anything 主张 **agent-native computer use**——结构化命令、显式状态、确定性反馈。
- **双入口：**
  1. **用现成生态：** `pip install cli-anything-hub` → `cli-hub list|search|install|launch`
  2. **生成新 harness：** 在 Claude Code / Pi / OpenClaw / Codex / Hermes 等平台装插件或 skill，执行 `/cli-anything <path-or-repo>` 跑满 7 阶段
- **7 阶段管线：** Analyze → Design → Implement（Click CLI + REPL + JSON + undo/redo）→ Plan Tests → Write Tests → Document（含 Phase 6.5 `SKILL.md`）→ Publish（`setup.py` / PATH）
- **产物形态：** 包名 `cli-anything-<software>`，命名空间 `cli_anything.*`；canonical skill 在仓库 `skills/`，亦可 `npx skills add HKUDS/CLI-Anything --skill …`
- **机器人/仿真相关示例 harness（仓内目录，非穷尽）：** `freecad`、`blender`、`godot`、`qgis`、`cloudcompare`、`renderdoc`、`nsight-graphics`、`comfyui` 等——服务 CAD、3D 资产、GIS、GPU 调试与生成式资产管线，而非替代 RL/控制栈。
- **质量叙事：** README 强调真实后端集成（非玩具重写）、单元 + E2E、Hub 注册与贡献流程（`CONTRIBUTING.md`）。

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/cli-anything.md`](../../wiki/entities/cli-anything.md)**
- 项目页归档见 [`sources/sites/cli-anything-hub.md`](../sites/cli-anything-hub.md)
- 技术报告归档见 [`sources/papers/cli_anything_arxiv_2606_03854.md`](../papers/cli_anything_arxiv_2606_03854.md)
- 交叉更新 Hermes / OpenClaw / Agent Reach / FreeCAD MCP 等关联页的「关联页面」节
