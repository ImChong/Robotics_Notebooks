# INSIGHT-Bench（Light Origins · 官方评测站 + 仓库 2026-09-21）

> 来源归档 — 项目页 <https://lightorigins.github.io/Light-INSIGHT-Bench/> + 仓库 README 核查

- **标题：** INSIGHT-Bench：Isaac Sim 上的诊断式 object-goal 导航评测
- **类型：** benchmark
- **项目页：** <https://lightorigins.github.io/Light-INSIGHT-Bench/>
- **代码：** <https://github.com/lightorigins/Light-INSIGHT-Bench>
- **数据集：** <https://huggingface.co/datasets/LightOriginsHQ/light-insight-bench>
- **Tech Blog（数据引擎背景）：** <https://www.lightorigins.com/blog/lightnav-0>
- **入库日期：** 2026-09-21
- **一句话说明：** 1097 episodes / 210 held-out 场景；5 场景类 × 5 指令类型矩阵；统一单目前向 RGB 协议；evidence-pack 可验证 leaderboard。

## 开源状态

- **评测侧已开源**（步骤 2.5，2026-09-21）：harness、1097-episode split、10 个可再分发 Habitat-GS 场景、7 基线 adapter。
- **训练侧部分公开**（博客口径）：1683 训练场景 / 53090 片段来自 HM3D/MP3D/InteriorGS/HabitatGS/VLNVerse 与五类指代；**未**随本仓库完整发布。

## 核心摘录

1. **5×5  taxonomy：** 场景类（Apartment / House / Commercial / Institution / Outdoor）× 指令类型（Base / Direction / Relation / Extremum / Ordinal）；行看布局敏感、列看语言机制、单元格看交互。
2. **部署协议（全模型统一）：** 480×270 前向 RGB、120° HFOV、1.0 m 高度；无 depth/odometry/panorama；300 actions；SR / SPL / terminal NE。
3. **Leaderboard 机制：** PR 提交 `submissions/*.json` 指向 public evidence pack；CI 校验 sha256 与 `insight-bench verify`；最多 5 episodes 可失败否则拒收。
4. **LightNav-0 论文表 vs 仓库复现：** README 给出 published 与 repository 两行 SR（Avg. 43.7 vs 44.9），强调采样随机性。

## 对 wiki 的映射

- [insight-bench](../../wiki/entities/insight-bench.md)
- [LightNav-0](../../wiki/entities/paper-lightnav-0.md)
- [3 篇技术地图](../../wiki/overview/lightorigins-3blogs-technology-map.md)
- [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)
