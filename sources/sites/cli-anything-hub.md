# CLI-Anything Hub（hkuds.github.io/CLI-Anything）

> 来源归档（ingest）

- **类型：** 网站 / 产品页 / 注册表前端
- **入口：** <https://hkuds.github.io/CLI-Anything/>（品牌域 <https://clianything.cc/>）
- **代码仓：** <https://github.com/HKUDS/CLI-Anything>
- **技术报告：** <https://arxiv.org/abs/2606.03854>
- **收录日期：** 2026-08-08
- **抓取说明：** 以 **2026-08-08** 对 Hub 公开 HTML 与仓库 README 交叉核对为准；注册表条目数与矩阵会随社区 PR 变化，勿在 wiki 固化具体数量。

## 一句话

**CLI-Hub** 是 CLI-Anything 的 **agent-friendly CLI 注册表与安装面**：浏览官方/社区生成的 harness 与部分第三方 Public CLIs，配合 `cli-anything-hub` PyPI 包与 `cli-hub-meta-skill` 让人类或代理 **发现 → 安装 → 调用**。

## 开源状态（步骤 2.5）

- **已开源：** Hub 前端与 registry（如 `registry.json` / `public_registry.json`）在主仓；安装器 `cli-anything-hub` 公开发布。
- **边界：** Hub 只负责发现与安装元数据；各 CLI 依赖的上游桌面/引擎软件（FreeCAD、Blender 等）版权与安装仍属原项目。

## 为什么值得保留

- 与 [仓库源归档](../repos/hkuds_cli_anything.md) 配对，区分 **生成方法论（HARNESS.md / 插件）** 与 **分发/发现面（Hub）**。
- 演示区直接展示与机器人工程相邻的 **FreeCAD Curiosity Rover**、**Blender hard-surface drone** 等 agent 轨迹产物，便于论证「CAD/3D 资产可 agent 化」。
- 提供 `npx skills add … cli-hub-meta-skill` 路径，衔接 OpenClaw / Claude Code / Codex 等本库已覆盖的宿主。

## 公开要点（归纳）

| 区块 | 内容 |
|------|------|
| **Empower agents** | 一键安装 Hub meta-skill；提示词引导代理自行 `cli-hub` 选型安装 |
| **Empower yourself** | `pip install cli-anything-hub`；`list/search/info/install/update/uninstall/launch` |
| **Matrices** | 能力包（intent → provider CLI）卡片视图；可 scoped 浏览 |
| **Collections** | CLI-Anything CLIs vs Public CLIs 两套目录 |
| **Demos** | FreeCAD / Blender / Draw.io 等真实产物 + preview / trajectory 叙事 |
| **贡献** | Contributing Guide + PR Template；wishlist / contributor signup（见 GitHub Issues 模板） |

## 对 wiki 的映射

- 升格页面：[wiki/entities/cli-anything.md](../../wiki/entities/cli-anything.md)
- 仓库侧归档：[sources/repos/hkuds_cli_anything.md](../repos/hkuds_cli_anything.md)
- 论文归档：[sources/papers/cli_anything_arxiv_2606_03854.md](../papers/cli_anything_arxiv_2606_03854.md)
