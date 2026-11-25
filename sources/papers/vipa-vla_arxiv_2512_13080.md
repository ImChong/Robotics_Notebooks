# Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos (VIPA-VLA)

> 来源归档（paper / arXiv:2512.13080v1，CVPR 2026）

- **URL：** <https://arxiv.org/abs/2512.13080>
- **PDF：** <https://arxiv.org/pdf/2512.13080.pdf>
- **项目页：** <https://beingbeyond.github.io/VIPA-VLA>
- **代码仓库：** <https://github.com/BeingBeyond/VIPA-VLA>
- **标题：** Spatial-Aware VLA Pretraining through Visual-Physical Alignment from Human Videos
- **作者：** Yicheng Feng, Wanpeng Zhang, Ye Wang, Hao Luo, Haoqi Yuan, Sipeng Zheng, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University、Renmin University of China
- **年份：** 2025/2026（arXiv 2025-12；CVPR 2026）
- **核对范围：** 已获取并通读 `arXiv:2512.13080v1` PDF 抽取全文（Sections 1–6，涵盖双视觉编码器 InternVL3.5-2B + Cut3R 3D 融合、Hand3D 数据构建 Pipeline、1024-bin 空间词表三阶段训练及 LIBERO/RoboCasa/Franka FR3 实验）；未独立复现模型训练与真机实验。
- **通读：** 已完成

## 摘要与核心贡献

1. **双路 2D+3D 视觉融合架构**：将语义编码器 **InternVL3.5-2B**（$V_{\text{sem}}$）与前馈 3D 点图重建编码器 **Cut3R**（$V_{\text{spa}}$）通过 Cross-Attention 与可学习残差门控参数 $\alpha$ 融合为空间感知视觉特征，兼顾高层语义理解与精确三维几何感。
2. **Hand3D 无机器人预训练数据管线**：完全从第一视角人类操作视频构建大规模三维图文对，包含两类数据：
   - **Hand3D-visual（300,368 条 VQA 样本）**：利用多视角相机标定、HaWoR 人手深度估计与深度图均值校准恢复绝对物理尺度，自动合成目标 3D 位姿、尺度对比与可操作性问答；
   - **Hand3D-action（1,030,850 条轨迹样本）**：将人手腕 3D 平动/旋转离散化为 1024-bin 空间动作词表作为物理空间对齐预训练目标。
3. **三阶段无真机预训练 + Flow-Matching 微调**：Stage 1 冻结骨干仅对齐 $V_{\text{sem}}/V_{\text{spa}}$ 融合层（Hand3D-visual）；Stage 2 解冻融合层与 LLM 在 Hand3D-action 上做空间动作对齐；Stage 3 接入 **DiT Flow-Matching 连续动作头**进行特定机器人下游微调。**预训练阶段完全不依赖机器人真机轨迹。**
4. **评测表现**：
   - LIBERO 双视角平均成功率 **96.8%**，单视角 **92.4%**；RoboCasa **45.8%**（超过 $\pi_{0.5}$ `41.4%`）；
   - 真机 Franka FR3 + Inspire Hand（Wipe-Board Seen / Unseen）：分别达 **83% / 60%**（对比 Being-H0 `40% / 10%`、GR00T N1.5 `57% / 30%`）。

## 与知识库关联

- 实体节点：[cn-os-vipa-vla.md](../../wiki/entities/cn-os-vipa-vla.md)
- 代码归档：[vipa-vla.md](../repos/vipa-vla.md)
