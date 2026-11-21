# UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer

> 来源归档（paper / arXiv:2512.21233v3）

- **URL：** <https://arxiv.org/abs/2512.21233>
- **PDF：** <https://arxiv.org/pdf/2512.21233.pdf>
- **项目页：** <https://beingbeyond.github.io/UniTacHand/>
- **代码仓库：** <https://github.com/BeingBeyond/UniTacHand>
- **标题：** UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer
- **作者：** Chi Zhang, Penglin Cai, Haoqi Yuan, Chaoyi Xu, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University
- **年份：** 2025（arXiv v3）
- **核对范围：** 已获取并通读 `arXiv:2512.21233v3` PDF 抽取全文（Sections 1–6 及附录，涵盖 MANO 1024×1024 UV 触觉映射、10 分钟配对采集集、线性插值轨迹增强、InfoNCE + 掩码重建 + GRL 域对抗联合表征学习及 RealMan + Inspire 真机迁移）；未独立复现硬件采集与真机测试。
- **通读：** 已完成

## 摘要与核心贡献

1. **MANO 规范 2D UV 空间触觉统一投影**：解决人类触觉手套（137 维压力通道 + 21 关键点动捕）与机器人灵巧手（Inspire Hand 1,062 维分布式压阻触觉矩阵 + URDF 运动学）传感器排布与构型异构问题，将双侧触觉读数均投影到 **1024×1024 MANO 手表面 2D UV 展开图**（经有效区域掩码与高斯空间平滑生成对齐触觉图 $U_H, U_R$，并可降维采样为 351 维税素向量用于下游 VLA/策略接口）。
2. **轻量配对表征对齐框架**：仅需 **10 分钟配对人类-机器人抓握交互数据**（50 种物体、688 条轨迹、40 Hz 采样共 16K 帧），采用线性轨迹插值数据增强与独立分区 CNN + MLP 编码器 $E_H, E_R$，联合优化：
   - 跨本体对称 **InfoNCE** 对比损失；
   - 随机掩码 UV 触觉图空间重建损失；
   - 梯度反转层（**GRL**）域对抗混淆损失，抹除人手与机械手在特征分布上的领域差异。
3. **零样本与单样本真机技能迁移**：在 **RealMan 6-DoF 机械臂 + Inspire DexHand** 上实现无需任何机器人触觉训练数据的**纯人类数据零样本策略迁移**：SoftHardPickPlace 成功率 **85%**、ObjectLocating **100%**、CompliantControl 柔顺控制 **40%**（显著优于 PatchMatch 基线 `10%`）；联合人机数据 1-shot 微调成功率达 **73.3%**（纯视觉基线仅 `43.3%`）。

## 与知识库关联

- 实体节点：[cn-os-unitachand.md](../../wiki/entities/cn-os-unitachand.md)
- 代码归档：[unitachand.md](../repos/unitachand.md)
- 后续工作：[cn-os-ttp.md](../../wiki/entities/cn-os-ttp.md)（沿用 UniTacHand 351 维 MANO UV 触觉空间构建 H-Tac 大规模预训练系统）
