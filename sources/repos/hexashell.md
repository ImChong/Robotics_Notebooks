# HexaShell（HexaFuture/HexaShell）

> 来源归档

- **标题：** HexaShell: A Thin Harness for Exploring the Limits of VLM Agents as Robot Coding Agents
- **类型：** repo / robot-coding-agent / manipulation
- **组织：** HexaFuture
- **代码：** <https://github.com/HexaFuture/HexaShell>
- **主页：** <https://hexafuture.ai>（官方 README 链接；本次无法读取站点正文）
- **论文：** <https://arxiv.org/abs/2610.23479>
- **评测：** [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo)
- **License：** Apache-2.0（仓库 LICENSE）
- **入库日期：** 2026-10-21
- **一句话说明：** 发布 HexaShell 在 RoboDojo 操纵任务中逐步构建并冻结的机器人工具、Open 任务共享配置和 EpisodeAPI 接口；不包含完整 Harness 服务器、Agent 容器或工具进化循环。

## 开源状态核查（2026-10-21）

| 项 | 值 |
|----|-----|
| **开放程度** | **部分开源** — 工具代码、接口、任务配置及评测结果公开；完整 Harness 与 Agent 执行 / 进化基础设施未包含 |
| **许可** | **Apache-2.0** |
| **核心模型** | 官方 README：Codex + GPT-6-Astra（medium reasoning effort） |
| **任务范围** | RoboDojo 42 个仿真任务；34 个任务使用各自工具，8 个 Open 任务共用一套配置 |
| **仿真环境** | 双臂 ARX X5 · RoboDojo · Isaac Sim |
| **官方评测** | README 报告成功率 **46.9%**、进度分 **53.2**，位列官方评测榜首；用户截图显示成功率 46.85% |
| **论文复测** | 新布局结果 **39.7% / 48.0**；与官方评测的布局及口径不同 |

## 仓库结构要点（README）

| 路径 | 作用 |
|------|------|
| `tasks/<task>/tools/<name>/` | 每项工具的 Python 实现、接口说明及辅助模块 |
| `tasks/<task>/enabled_tools.txt` | 各任务与随机变体启用的工具 |
| `tools/table_plane/`、`tools/pick/` | 多视角桌面高度测量和抓取工具 |
| `tools/scene/`、`library/`、`profiles/open/` | 8 个 Open 任务共享的视觉工具、技能与配置 |
| `interface/EpisodeAPI.md` | 工具可调用的 Episode API 契约 |
| `recipes/`、`results/` | 任务说明及各轮评测结果 |

当前仓库包含 34 个任务的 196 个工具；官方评测使用其中按任务冻结的工具集。仓库 README 明确说明完整 Harness（server、executors、Agent container、evolution loop）不在本仓库。

## 复现注意事项

工具需由兼容的 Harness 调用 `run(api, command, args)`，并运行 RoboDojo Isaac Sim 环境。仅克隆此仓库不能直接启动完整 Agent 评测；复现者需提供 EpisodeAPI 实现与评测运行时。项目报告的榜单结果由 RoboDojo 官方评测执行，不应与论文在作者机器上的新布局复测混为一谈。

## 对 wiki 的映射

- 主实体：[HexaShell](../../wiki/entities/paper-hexashell.md)
- 基准实体：[RoboDojo](../../wiki/entities/robodojo.md)
- 论文摘录：[hexashell_arxiv_2610_23479.md](../papers/hexashell_arxiv_2610_23479.md)
