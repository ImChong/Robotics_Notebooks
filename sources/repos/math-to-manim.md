# Math To Manim（HarleyCoops/Math-To-Manim）

- **标题**：Math To Manim
- **类型**：repo
- **来源**：HarleyCoops（Christian H. Cooper）
- **链接**：<https://github.com/HarleyCoops/Math-To-Manim>
- **克隆**：`https://github.com/HarleyCoops/Math-To-Manim.git`
- **入库日期**：2026-10-11
- **一句话说明：** 开源数学动画创作与审查管线：把自然语言问题编排为教学简报、数学推导、分镜和 Manim 场景，再进行源码检查、实际渲染和画面审阅。
- **开放状态：** 已开源；官方 GitHub 仓库公开包含源码与安装说明，许可证为 MIT。该仓库 README 是主要项目说明与安装入口。
- **沉淀到 wiki：** 是 → [Math To Manim 实体页](../../wiki/entities/math-to-manim.md)；底层渲染引擎见 [Manim](../../wiki/entities/manim.md)。

## 仓库概况（2026-10-11 README / pyproject.toml）

| 字段 | 值 |
|------|-----|
| 托管 | GitHub，默认分支 `main` |
| 主要语言 | Python |
| Python | `>=3.10` |
| 许可证 | MIT |
| 渲染引擎 | Manim（可选 render extra） |
| 主要入口 | `math-to-manim` / `m2m` 命令行；Astra MCP server |
| 当前主流程 | Codex SDK + GPT-6 Astra；Jev 可选审查 |
| 系统渲染依赖 | FFmpeg、LaTeX / `dvisvgm` 与 Manim 所需系统包 |

## README 与实现摘要

### 当前 Astra 主流程

当前主命令 `math-to-manim` / `m2m` 调用 Astra 管线。README 将角色划分为：Astra 形成学习简报、核查数学、设计视觉论证、编写 Manim 场景，并检查实际渲染结果。默认情况下，TypeSafe Jev 的阶段评审为**建议式**，记录分数而不自动触发重生成；可选择严格门禁，或用 `--review-mode off` 跳过 Jev。Jev 评审不是数学正确性的形式证明。

一次运行会保留提示词、候选版本、评审记录、Manim 场景源码、渲染日志、抽样帧和完成清单；生成文件存放在当前工作目录的 `runs/astra/<run-id>/`。因此可检查的不只是最终 MP4，也包括生成过程中的素材和审核证据。

### 渲染与接口

- **Manim 才是实际动画引擎。** Math To Manim 负责把自然语言问题变成教学设计、场景代码和审查/渲染流程；它并未替代 Manim。
- 主 Astra 入口提供 CLI 和 MCP。MCP 可启动 Astra/Jev 动画任务，并读取任务、运行记录、场景源码和产物。
- 仓库还保留 Mythos 等旧版 provider pipeline；其中 Mythos 提供 FastAPI REST API（`math-to-manim serve-api`）。不同入口对应不同管线，不应把旧服务接口误认为 Astra 默认链。
- 上游 Astra bundle 可通过发布包安装；从源码安装则需 Python、Node.js / npm、Manim 系统依赖、FFmpeg 与 LaTeX。README 提醒 `MathTex` / `Tex` 还要求 LaTeX 模板所需的包；单纯安装二进制不足以保证公式渲染成功。

## 对机器人研究与工程的关联

- 可将控制、优化、强化学习、机器人几何等数学概念制作成可复现讲解片，适用于课程、组会和项目演示等**知识传播层**。
- 逐阶段保留场景源码与画面证据，适合作为教学内容制作与审阅的工具链样本；核心复现判断仍需检查生成的公式、代码与具体帧。
- **不属于机器人仿真或策略训练工具。** 生成视频中的几何/运动由 Manim 场景定义，不等价于物理仿真、机器人动力学验证或真实硬件回放。
- 上游当前主流程使用模型与 Codex 登录；开源代码不代表每种运行路径都离线或免凭据。关闭 Jev 可免其单独凭据要求，但仍保留 Astra 生成与审查调用。

## 对 wiki 的映射

- [Math To Manim 实体页](../../wiki/entities/math-to-manim.md)
- [Manim 动画引擎](../../wiki/entities/manim.md)
- [arXivisual](../../wiki/entities/arxivisual.md) — 论文阅读器中嵌入 Manim 动画的相邻应用

## 参考链接

- 上游仓库与 README：<https://github.com/HarleyCoops/Math-To-Manim>
- 项目配置 / Python 版本 / 许可证：<https://github.com/HarleyCoops/Math-To-Manim/blob/main/pyproject.toml>、<https://github.com/HarleyCoops/Math-To-Manim/blob/main/LICENSE>
- Astra 管线说明：<https://github.com/HarleyCoops/Math-To-Manim/blob/main/docs/ASTRA_PIPELINE.md>
- Manim 安装指南：<https://docs.manim.community/en/stable/installation/uv.html>
