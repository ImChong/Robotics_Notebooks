# Co-VLA: Coordination-Aware Structured Action Modeling for Dual-Arm Vision-Language-Action Systems

> 一手论文归档（paper）

- **中文标题：** Co-VLA：面向双臂 VLA 的协同感知结构化动作建模
- **类型：** paper / bimanual / coordination / vla
- **arXiv：** https://arxiv.org/abs/2606.20285
- **HTML：** https://arxiv.org/html/2606.20285v1
- **PDF：** https://arxiv.org/pdf/2606.20285
- **版本：** arXiv v1（2026-06）
- **官方代码：** 本次原文核对未见可运行官方仓库或官方项目源码入口。
- **入库日期：** 2026-11-21
- **核对范围：** arXiv v1 HTML 的 Structured Action Expert、共享/残差协同表示、损失项、LAC 执行时约束、Co-Motion 数据采集、RoboTwin 2.0 及 Agilex 真机评测、消融和限制。
- **通读：** 已完成

## 核心摘录

1. Structured Action Expert（SAE）把双臂动作分解为共享协同分量与各臂残差分量，利用结构先验提升跨臂时序同步，不仅依赖视觉隐层自行发现左右臂关系。
2. 训练中加入角色/结构、稀疏与时序协同的辅助正则，论文主实验使用量级很小的辅助权重（报告的 λ=0.001），避免覆盖原始动作生成损失。
3. Latent Action Correction（LAC）在推理/执行端对联动动作进行约束修正与平滑，补偿预测块与双臂同步/安全要求；不能将此等同可证明碰撞安全控制。
4. RoboTwin 2.0 Easy/Hard 评估报告 Co-VLA 平均成功率约 82%/22%，π0 参照约 76%/21%；各任务差异明显，例如 Handover Block Easy 91% vs 64%，不应只报平均。
5. Agilex 双臂真机任务以每任务约 50 示范训练、不同设定分别 30 次执行评估；LAC 并非所有 OOD 子任务均提升，需保留有无修正器的逐项结果。
6. Co-Motion 使用同步协同数据采集，在示范效率与学习成功率间存在取舍；不能把数据采集提速直接等同策略成功率提升。

## 局限与工程前提

- 对双臂机械同步、碰撞边界和任务配对的数据分布有依赖；复杂接触外推不足。
- 对照数据、单臂和双臂动作空间必须分开，不应与 H-VLA 的关键动作层次控制混淆。
- 论文当前未给出可运行官方代码入口，源码时序图不适用。

## 对 wiki 的映射

- [Co-VLA 唯一论文节点](../../wiki/entities/paper-co-vla.md)
- [接触丰富操作纵深](../../roadmap/depth-contact-manipulation.md)
