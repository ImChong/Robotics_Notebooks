# OMOMO（Object Motion Guided Human Motion Synthesis）

- **标题**: OMOMO Dataset & Code
- **类型**: dataset / repo
- **仓库**: https://github.com/lijiaman/omomo_release
- **项目页**: https://lijiaman.github.io/projects/omomo/
- **论文**: Li et al., *Object Motion Guided Human Motion Synthesis*, SIGGRAPH Asia 2023 (TOG) — arXiv:[2309.16237](https://arxiv.org/abs/2309.16237)
- **数据下载**: Google Drive（README 链接；需 SMPL-H / SMPL-X 模型）
- **收录日期**: 2026-06-16

## 一句话摘要

Stanford 团队发布的人–物交互（HOI）动捕数据集与条件扩散合成代码：**15 类物体、约 10 小时** 全身操纵动作，含 3D 物体几何、物体运动与人类全身姿态；下游常被 GMR / OmniRetarget 等用作 **loco-manipulation 重定向源**。

## 为何值得保留

- **交互语义明确**：每条序列绑定具体物体运动与人类全身姿态，适合 **搬运、推拉、持物行走** 等人形 loco-manipulation 研究。
- **机器人管线常见上游**：OmniRetarget、ResMimic、VisualMimic、holosoma 等公开工作将 OMOMO 与 AMASS / LAFAN1 并列作为 MoCap 源。
- **与纯 locomotion 库互补**：AMASS / LAFAN1 偏全身运动；OMOMO 强调 **物体状态 → 人体操纵** 的条件关系。

## 数据要点（编译自项目页与 README）

| 字段 | 内容 |
|------|------|
| 规模 | 15 物体 · 约 **10 h** 交互动作 |
| 表示 | SMPL-H / SMPL-X 系人体姿态 + 物体几何与运动 |
| 许可 | 遵循 SMPL-H / SMPL-X 及数据集自身条款（使用前需注册下载人体模型） |
| 代码 | 两阶段扩散（手部位姿 → 全身）+ 训练/测试脚本 |

## 对 Wiki 的映射

- **wiki/entities/omomo-dataset.md**：数据集实体页（归纳级）。
- **wiki/comparisons/humanoid-reference-motion-datasets.md**：与 AMASS / LAFAN1 / PHUMA / Humanoid Everyday 选型对照。
- **wiki/entities/omniretarget-dataset.md**：OmniRetarget HF 子集 `robot-object/` 的 OMOMO 来源互链。
