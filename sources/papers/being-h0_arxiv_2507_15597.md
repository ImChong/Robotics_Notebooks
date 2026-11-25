# Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos

> 来源归档（paper / arXiv:2507.15597v1，ICML 2026）

- **URL：** <https://arxiv.org/abs/2507.15597>
- **PDF：** <https://arxiv.org/pdf/2507.15597.pdf>
- **项目页：** <https://research.beingbeyond.com/being-h0>
- **代码仓库：** <https://github.com/BeingBeyond/Being-H0>
- **标题：** Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos
- **作者：** Hao Luo, Yicheng Feng, Wanpeng Zhang, Sipeng Zheng, Ye Wang, Haoqi Yuan, Jiazheng Liu, Chaoyi Xu, Qin Jin, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Peking University、Renmin University of China
- **年份：** 2025/2026（arXiv 2025-07；ICML 2026）
- **核对范围：** 已获取并通读 `arXiv:2507.15597v1` PDF 抽取全文（Sections 1–6 及附录，涵盖 UniHand 166.5M 数据集、Part-level GRQ 手部运动分词器、Physical Space Alignment 相机内参/尺度归一化、InternVL3 骨干预训练及非自回归真机后训练对齐）；未独立复现百万级视频预训练与 Franka 真机硬件部署。
- **通读：** 已完成

## 摘要与核心贡献

1. **UniHand 大规模人手操作语料**：汇集 11 个公开第一视角/多视角人手-物体交互数据集构建 **166.5M 样本、1,155 小时**（及精选高质量子集 **UniHand-2.5M**），统一参数化为 **MANO-D162**（手腕平动+6D旋转、手指局部关节旋转及 21 个 3D 关键点）。
2. **Part-level Grouped Residual Quantizer（GRQ）**：将手腕 3D 平动/旋转与手指 articulation 解耦为独立分组残差量化分支（每支码本 4096，合计 `Being-H0-GRVQ-8K`），以 15 FPS 采样率编码为 `128 tokens/s` 离散运动块 `<MOT>...</MOT>`；训练期采用**词表级 Logit Masking** 与**双百分位 Token Loss Masking**（过滤静止平凡帧与标注噪声抖动）。
3. **Physical Space Alignment（物理空间对齐）**：解决多源视频相机焦距、分辨率差异带来的 3D 尺度歧义——包含弱透视缩放-裁切对齐（Unified Intrinsics）与随机平面内旋转增强，并支持 Free / Block / Soft 三种解码模式。
4. **真机动作分块对齐后训练**：在 Franka FR3（7-DoF 机械臂）+ Inspire 灵巧手（6-DoF）平台上，将骨干结构由自回归改为**可学习 Action Query + 双向注意力 + 连续 MLP Action Chunk L1 回归头**，提供 `Being-H0-{1B,8B,14B}-2508` 及真机微调权重。

## 与知识库关联

- 实体节点：[paper-notebook-being-h0-vision-language-action-pretraining-from.md](../../wiki/entities/paper-notebook-being-h0-vision-language-action-pretraining-from.md)
- 代码归档：[being-h0.md](../repos/being-h0.md)
- 后续演化：`Being-H0.5`（跨本体 MoT，见 `cn-os-being-h.md`）、`Being-H0.7`（[being-h07.md](../../wiki/methods/being-h07.md)）
