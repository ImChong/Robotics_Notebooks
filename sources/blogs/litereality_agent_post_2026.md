# LiteReality-Agent 官方博客（2026）

> 来源归档（ingest）

- **标题：** LiteReality-Agent: An Agentic System for Interactable 3D Indoor Scene Reconstruction
- **类型：** blog / technical-report / real2sim / 3d-reconstruction
- **URL：** <https://litereality.github.io/Litereality-agent-site/litereality-agent-post/>
- **PDF：** <https://litereality.github.io/Litereality-agent-site/litereality-agent-post/litereality-agent.pdf>
- **代码：** <https://github.com/LiteReality/LiteReality-Agent>
- **机构：** 剑桥大学（Joan Lasenby、Shangzhe Wu 等；以 PDF 为准）
- **入库日期：** 2026-09-08
- **一句话说明：** iOS LiDAR 扫描 + 确定性 scene init + agent 编辑单一 `Room.py` 的 realism authoring；QC 门控后导出可交互室内场景；Technical Report 标注 coming soon。

## 开源状态（步骤 2.5，2026-09-08）

| 组件 | 状态 |
|------|------|
| GitHub | **已开源** — `uv run litereality` CLI、Modal 部署脚本、example-scans |
| Scanner App | **已发布** — App Store 免费 |
| Technical Report PDF | 项目页已挂 PDF；README 仍写 coming soon |

**结论：已开源** — 管线与 App 可跑；正式 TR 与大规模 stress-test 仍进行中。

## 核心摘录

### 摘录 1：两阶段管线

- **Scene init（确定性）：** RoomPlan USDZ → 逐物体 TRELLIS 或 Articraft 式程序化分支 → 按扫描位姿组装 seed room。
- **Authoring（agentic）：** 代理主要编辑 **单一 `Room.py`（Blender Python）**；工具含 `select_view`、`render_and_compare`、`grid`、`critic`、`fetch_materials`；固定 tool-call 预算迭代。
- **QC gate：** 碰撞、几何、材质、关节等确定性检查 + 末段 model checklist；通过后才导出。

**对 wiki 的映射：** [litereality-agent](../../wiki/entities/litereality-agent.md)

### 摘录 2：动机与局限

- 动机：熟悉空间的 **可编辑可交互** 重建；以及 **scan→interactive scene→（未来）sim-ready** 降低 reality-grounded sim 成本。
- 局限：**尚未 simulation-ready**（整房间进 MuJoCo/Isaac 仍需大量工程）；当前 stress-test 以 **≤50 m² 单房间** 为主。

**对 wiki 的映射：** [litereality-agent](../../wiki/entities/litereality-agent.md)

## 当前提炼状态

- [x] 项目页、Blog、GitHub README 核查（2026-09-08）
- [x] wiki 映射：`wiki/entities/litereality-agent.md`
