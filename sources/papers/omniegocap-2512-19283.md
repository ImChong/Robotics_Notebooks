# OmniEgoCap：设备无关的序列级第一人称运动重建

> 来源归档（paper）

- **论文：** <https://arxiv.org/abs/2512.19283>；核对版本 [v3](https://arxiv.org/html/2512.19283v3)，2026-10-19。
- **项目页：** <https://kyungwoncho.github.io/OmniEgoCap/>
- **代码：** <https://github.com/KyungWonCho/OmniEgoCap>
- **作者与机构：** Kyungwon Cho、Jeonghyeon Na、Hanbyul Joo；首尔大学。
- **会议：** NeurIPS 2026
- **入库日期：** 2026-11-04

## 核对摘录

1. §3.1–3.2：输入为世界系头部轨迹及间歇可见的双腕 6 自由度位姿；需相机到头部标定。输出 SMPL-H 体型、根位姿和关节旋转。
2. §3.3–3.5：序列级扩散结合全局体型与可见性上下文；训练中通过相机几何变化合成手部可见性。
3. §4.3：五种模拟视场配置下，论文表 2 报告无优化版本的全身 MPJPE 为 93.65、85.70、72.22、75.70、98.00 mm；手部误差并非所有条件下均最优。
4. §4.4：OmniEgoDB 包含 4 人、50 段、177 分钟及 Xsens 参考；表 3 无优化版本 Aria stereo / RGB、GoPro、Quest 的 MPJPE 分别为 106.61 / 118.11 / 111.68 / 110.49 mm。

## 核对范围

- **通读：** 未完成

已核对 arXiv v3 摘要、问题定义、预处理、部分实验与附录，以及官方项目页和推理入口；**尚未通读全文**，待补读其他实验、局限及全部相关附录。

## 对 wiki 的映射

- [OmniEgoCap 实体](../../wiki/entities/paper-sa-2512-19283-omniegocap-camera-agnostic-sequence-level-egocen.md)
- [官方项目页归档](../sites/omniegocap.md)
- [官方代码归档](../repos/omniegocap.md)
