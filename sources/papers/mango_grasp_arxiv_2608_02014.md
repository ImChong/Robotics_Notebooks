# MANGO-Grasp: Mahalanobis Fields over Geometry-Oriented 3D Gaussians（arXiv:2608.02014）

> 来源归档（ingest）

- **标题：** MANGO-Grasp: Mahalanobis Fields over Geometry-Oriented 3D Gaussians for Cross-Embodiment Dexterous Grasping
- **缩写 / 框架：** **MANGO-Grasp**
- **类型：** paper / dexterous-grasping / cross-embodiment / 3dgs
- **arXiv：** <https://arxiv.org/abs/2608.02014>
- **项目页：** <https://connor-zh.github.io/MANGO-Grasp/>（归档见 [`sources/sites/mango-grasp.md`](../sites/mango-grasp.md)）
- **作者：** Heng Zhang、Kevin Yuchen Ma、Mike Zheng Shou、Weisi Lin、Yan Wu∗
- **机构：** 新加坡科技研究局资讯通信研究院（A*STAR-I2R）；南洋理工大学（NTU）；新加坡国立大学 Show Lab（NUS）
- **入库日期：** 2026-08-15
- **一句话说明：** 物体用几何导向的板状 3D Gaussian（外法向），手用表面关键点的形态–运动学描述子；关键点–基元上的马氏场作训练目标与推理优化引导。同一套优化超参跨手型；未见 SharpaWave 零样本，真机成功率 **86%**。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-15）：** [connor-zh.github.io/MANGO-Grasp](https://connor-zh.github.io/MANGO-Grasp/) 有方法图、真机 10 物演示与 86/100 表；按钮写 **Code / Data & Checkpoints「released upon publication」**。
- **结论：** **宣称将开源。** 源码运行时序图标 **不适用**。

## 摘录 1：表示

| 侧 | 设计 |
|----|------|
| **物体** | \(G=256\) 表面对齐板状 Gaussian \((\mu,R,\sigma,n)\)；用网格法向监督做几何驱动 densify，按曲率分配容量 |
| **手** | \(N=256\) 表面关键点；预训练同时学形态身份与跨构型运动 |
| **交互** | 马氏场 \(\hat M\in\mathbb{R}_+^{N\times G}\)：沿法向陡、切向缓；宽平面基元横向更宽容 |

## 摘录 2：实现

预测场引导从 \(q^0\) 优化到 \(q^*\)，加关节限位、穿透与自碰能量。**所有 embodiment 共用同一优化公式与超参。**

## 摘录 3：数字

- 见手（Shadow / Allegro / Barrett）：CMAP / MultiGripperGrasp 仿真成功率 **97.59% / 89.47%**；相对最强见手基线最高 **+8.24 pp**。
- 未见 SharpaWave 零样本：**84.17% / 81.47%**；相对最强零样本基线最高 **+16.57 pp**。
- 真机未见手、无微调：10 物 ×10 次，平均 **86%**（海绵/番茄汤罐/牙膏盒 10/10；苹果 6/10）。

**对 wiki 的映射：** [`wiki/entities/paper-mango-grasp.md`](../../wiki/entities/paper-mango-grasp.md)；交叉 [抓取位姿估计](../../wiki/methods/grasp-pose-estimation.md)、[UHAS](../../wiki/methods/uhas-unified-hand-action-space.md)、[灵巧手运动学](../../wiki/concepts/dexterous-kinematics.md)、[DigitCode](../../wiki/entities/paper-digitcode.md)。

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 开源状态核查（出版后开源）
