# Granular Terrain Humanoid（arXiv:2609.10286）

> 来源归档（ingest）

- **标题：** 面向颗粒地形的人形机器人地形自适应运动学习
- **英文标题：** Learning Terrain-Adaptive Humanoid Locomotion on Granular Terrain
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.10286>
- **PDF：** <https://arxiv.org/pdf/2609.10286>
- **开源：** 截至入库日 **未见** 官方仓库（步骤 2.5：项目页/arXiv 未给出可运行代码链接）。
- **入库日期：** 2026-09-14
- **策展索引：** [wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md](../blogs/wechat_shenlan_weekly_humanoid_quadruped_2026-09-14.md)

## 核心论文摘录

### 1) 3D 阻力理论颗粒接触求解器

- 仿真中可解析颗粒反力，支撑 Teacher 在特权地形上训练。
- **对 wiki 的映射：** [../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md](../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md)

### 2) Teacher-Student + VAE 特权地形

- Student 仅用 onboard 传感，VAE 压缩地形潜变量。
- **对 wiki 的映射：** [../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md](../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md)

### 3) 零样本迁移真实颗粒

- 玄武岩、干沙、海滩沙未再训练直接部署。
- **对 wiki 的映射：** [../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md](../../wiki/entities/paper-granular-terrain-humanoid-locomotion.md)

## 步骤 2.5 开源核查

- 已检索 arXiv 摘要与常见项目页关键词（GitHub/code）；**未发现**可运行官方实现。
- 若后续发布代码，应同步 `sources/repos/` 与本 wiki 页「工程实践」与「源码运行时序图」。

## 当前提炼状态

- [x] 公众号周更 ingest 映射
- [x] wiki 实体页
