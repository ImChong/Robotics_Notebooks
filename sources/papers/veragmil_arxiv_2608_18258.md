# VERAGMIL（arXiv:2608.18258）

> 来源归档（ingest）

- **标题：** VERAGMIL: Virtual Environment for Scooping Granular Foods with Imitation Learning Models
- **类型：** paper / assistive-robotics / imitation-learning / vr-simulation / granular-manipulation
- **arXiv abs：** <https://arxiv.org/abs/2608.18258>
- **PDF：** <https://arxiv.org/pdf/2608.18258>
- **代码：** <https://github.com/AmanuelErgogo/VERAGMIL>（归档见 [`sources/repos/veragmil.md`](../repos/veragmil.md)）
- **会议：** IROS 2025
- **机构：** SANO Centre（Kraków）；南佛罗里达大学（USF）；维罗纳大学（University of Verona）
- **作者：** Amanuel Ergogo、Diego Dall'Alba、Przemyslaw Korzeniowski
- **发表 / 上传：** 2026-08-21（arXiv v1）
- **入库日期：** 2026-08-21
- **索引来源：** [具身智能小站 8 篇综述](../blogs/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)（<https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g>）

## 开源状态（步骤 2.5，2026-08-21）

- **待发布：** README 描述 install/train/eval 流程并引用 `scripts/`、`env/`、`weights/bcq.ckpt`，但仓内 **仅 README + demo GIF**，无代码/配置/权重。
- **结论：** 论文称 framework public；截至入库日 **不可运行复现**。

## 摘录 1：问题

- 助残喂食需舀取/运输米饭、豆类等 **颗粒食物**；材料动态复杂，高质量人类示范难采。

## 摘录 2：VERAGMIL 环境

- **高保真 Isaac Sim 仿真** + 直观 **VR**（Quest 2/3）示范界面；xArm7 + 多类物理特性食物。
- 训练 **BC / BC-RNN / BCQ**；按成功率、**洒落量**、**未见食物泛化**、完成时间评估。

## 摘录 3：结果

- **VR 示范显著优于 3D 空间鼠标**；**BCQ** 综合最好，尤其减少洒落并接近人类表现。

**对 wiki 的映射：** [`wiki/entities/paper-veragmil.md`](../../wiki/entities/paper-veragmil.md)；交叉 [模仿学习](../../wiki/methods/imitation-learning.md)、[Isaac Gym / Isaac Lab](../../wiki/entities/isaac-gym-isaac-lab.md)。

## 当前提炼状态

- [x] GitHub shell 仓核查
- [x] 升格 `wiki/entities/paper-veragmil.md`
