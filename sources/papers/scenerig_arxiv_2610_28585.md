# SceneRig（arXiv:2610.28585）

> 来源归档（paper）

- **标题：** SceneRig: An Agentic System for Simulation-Ready 3D Scene Reconstruction from Single Images
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.28585>
- **PDF：** <https://arxiv.org/pdf/2610.28585>
- **项目页：** <https://red-fairy.github.io/SceneRig/>
- **代码：** <https://github.com/Red-Fairy/SceneRig>
- **入库日期：** 2026-10-31
- **线索来源：** [一张照片变成仿真场景，机器人真能用吗？｜附代码与项目入口](../blogs/wechat_embodied_station_50_papers_scenerig_photo_to_sim_2026-10-31.md)（具身智能小站，2026-10-31）
- **一句话说明：** 真实场景搬进仿真后，物体看起来相似，却可能一碰就倒、抓取位置错位。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-31）：仓库可达（git ls-remote）。项目页 HTTP 200。

## 核心摘录（公众号转述，未核验原文表格）

- 让照片重建经得起机器人操作。SceneRig把几何对齐与物理验证纳入重建循环，并进一步检查能否预测真实策略表现、复现操作轨迹，适合关心仿真评测与数据生产的团队深读。
- 真实场景搬进仿真后，物体看起来相似，却可能一碰就倒、抓取位置错位。
- 指标：公众号转述：五项任务、两种策略共 100 次真实回合的对应仿真中，成败一致率 SceneRig 为 69%、SimFoundry 为 54%（非机器人任务成功率）；25 条成功真机示范直接回放，SceneRig 完成 20/25、SimFoundry 为 9/25（重建均使用标定深度和相机参数）；当前主要面向桌面刚体场景，尚不恢复物体关节与形变。

## 对 wiki 的映射

- [paper-scenerig](../../wiki/entities/paper-scenerig.md)
- [单图重建仿真、VLA 适配评测与跨形态世界模型：具身智能小站 10-31 期 50 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-31-50-papers-technology-map.md)
