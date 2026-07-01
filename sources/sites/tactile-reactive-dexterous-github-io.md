# tactile-reactive-dexterous.github.io（T-Rex 项目页）

- **标题：** T-Rex: Tactile-Reactive Dexterous Manipulation
- **类型：** site / project-page
- **URL：** <https://tactile-reactive-dexterous.github.io/>
- **入库日期：** 2026-07-01
- **配套论文：** [T-Rex（arXiv:2606.17055）](https://arxiv.org/abs/2606.17055) — 归档见 [`sources/papers/trex_arxiv_2606_17055.md`](../papers/trex_arxiv_2606_17055.md)
- **代码：** <https://github.com/ZhuoyangLiu2005/T-Rex>

## 一句话摘要

T-Rex 官方项目页：展示 **100 小时开源触觉数据集** 统计与浏览器可视化器、**变频率 MoT + 异步触觉 flow matching** 架构动画、**12 项真机自主 rollout** 视频，以及相对 **EgoScale / π₀.₅ / Tactile-VLA** 等基线的成功率对比与消融。

## 公开信息要点（截至入库日）

- **核心叙事：** 触觉反应式灵巧操作需要 **数据（T-Rex Dataset）+ 架构（MoT 双时钟）+ 训练配方（人预训练 → 触觉 mid-training → 任务微调）** 三者同时到位；朴素把触觉拼进 VLA **会伤害性能**。
- **数据集板块：** 开源 **50 h** 遥操作 play（项目页口径；论文叙述合计 **100 h** 含未全开源部分）；**200+** 物体、**22** 运动基元；提供 **500 轨迹随机子集** 交互可视化器。
- **演示板块：** Flip Page、Transfer Egg、Wipe Plate、Apply Toothpaste、Split Cup、Sort Mahjong、Open Lock、Acid-Base Neutralization、Extract Card、Screw Lightbulb 等 **12** 任务真机视频。
- **结果板块：** T-Rex **65%** 宏平均成功率，**EgoScale 35%**；逐任务表、触觉模态/异步级联/训练阶段消融、数据效率曲线。
- **失败案例：** 物体碰撞、滑脱、位姿偏差、多指摩擦、过力、滑动不对齐六类典型模式。
- **致谢：** 项目页基于开源 **ENPIRE** 模板构建。

## 为何值得保留

- **非 PDF 证据：** 异步级联 rollout 动画与逐任务视频比表格更直观呈现 **高频触觉 tick 如何叠在低频 visuomotor chunk 上**。
- **与论文三角互证：** 项目页数值与 arXiv Table 1、Figure 5–7 一致，便于维护者核对表述。

## 关联资料

- 论文归档：[`sources/papers/trex_arxiv_2606_17055.md`](../papers/trex_arxiv_2606_17055.md)
- 代码归档：[`sources/repos/t-rex-zhuoyangliu.md`](../repos/t-rex-zhuoyangliu.md)
