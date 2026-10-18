# Murea-InWorld（murea.ai）

> 来源归档（ingest · 项目页）

- **标题：** Murea-InWorld
- **类型：** site / project-page
- **官方入口：** <https://murea.ai/>
- **技术报告：** <https://murea.ai/page/paper/InWorld-technical-report.pdf>
- **代码：** <https://github.com/Murea-Labs/InWorld>
- **权重：** <https://huggingface.co/MureaLabs/InWorld>
- **机构：** Murea Labs
- **入库日期：** 2026-10-18
- **一句话说明：** 面向文字驱动互动故事的视听世界模型：Agent 维护世界状态并规划事件，视频生成器据此生成带同步声音的片段。

## 开源状态

官方 GitHub README 将 Murea-Mind（故事 Agent）和 Murea-Fast（推理服务）列为仓库提供的组件，并链接到 Hugging Face 的 Murea-Fast 权重；部署说明还要求 LTX-2.5 基础组件。技术报告摘要包含 Murea-Pro 与 Murea-Fast 的评测结果，但 README 没有将 Murea-Pro 列为开源组件。**结论：Murea-Mind 与 Murea-Fast 推理代码已开源，Murea-Fast 权重已发布；Murea-Pro 开源状态未确认。**

项目主页和报告链接由官方 GitHub README 列出；当前抓取渠道未能读取 murea.ai 正文，故不将项目页上未核实的内容作为依据。

## 技术报告评测摘录（2026-10-18 复核）

- **链接：** 技术报告 PDF §5–§6（Table 1–3）
- **摘录要点：**
  - WBench Full-289（289 例、1,058 轮交互）：Murea-Pro Overall 81.13，高于对比中的 Kling 3.0（79.42）；优势集中在 Interaction（82.60 vs 73.08）与 Physical（74.42 vs 69.19），Setting / Consistency 低于 Kling 3.0。
  - OpenS2V-Eval：Murea-Fast Total 64.98，高于所评开源基线 Bernini（63.21）与 MiniMax-H3（61.01）；Seedance 2.0 为 67.13 但有 32 例拒答，148 个共有样例上为 66.11 vs 67.13。
  - OS2V-Pro（报告自建扩展评测）：Murea-Fast 聚合 69.9，Kling V3 Omni 70.1。
  - 延迟：8 步 Murea-Fast 在 8×H20 上生成 5.04 秒片段 43.7 秒。
  - §6 自述短板：导航、长程规划与场景设定一致性。
- **对 wiki 的映射：**
  - [Murea-InWorld 实体页](../../wiki/entities/paper-murea-inworld.md) — 「实验与评测」「结论」节。

## 对 wiki 的映射

- [Murea-InWorld 实体页](../../wiki/entities/paper-murea-inworld.md)
- [生成式世界模型](../../wiki/methods/generative-world-models.md)
- [视频作为仿真](../../wiki/concepts/video-as-simulation.md)

## 当前提炼状态

- [x] 官方仓库 README、LICENSE 和 Hugging Face 权重链接核查
- [x] 技术报告正文评测部分已直接复核
- [ ] 项目主页正文待直接复核
