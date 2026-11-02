# PDRVLA（arXiv:2610.37381）

> 来源归档（paper）

- **标题：** PDRVLA: A Vision-Language-Action Model with Physical Dynamics Representations
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.37381>
- **PDF：** <https://arxiv.org/pdf/2610.37381>
- **作者：** Wang, Hanqing, Cai, Zhaotian, An, Ruichuan, Zeng, Bohan, Zhang, Tao, Chen, Yilin, Jiang, Xu, Chen, Kaiquan et al.
- **arXiv 评论栏：** （无）

- **入库日期：** 2026-11-02
- **一句话说明：** PDRVLA：具有物理动力学表示的视觉语言动作模型。

## 开源状态

- arXiv 摘要页与评论未见代码或项目页链接（仅核对 arXiv 摘要页，不代表未发布）（2026-11-02，仅核对 arXiv 摘要页）

## 核心摘录

1. **策展来源：** [【10.26-10.30前沿论文动态】Manipulation 57篇](../blogs/wechat_senlanke_weekly_manipulation_2026-10-26_30.md)
2. **周报所述问题：** VLA 缺少对环境物理变化的有效建模，而在控制循环中直接生成未来视频会增加推理延迟。
3. **周报所述创新点：** 提出轻量化 Physical Dynamics Module，从视觉语言特征中提取物理 Token，并预测未来 RGB、深度和光流特征。训练时使用冻结的视频世界模型提供辅助监督，推理时无需生成未来视频。在 LIBERO、LIBERO-Plus、RoboTwin 2.0 和 SimplerEnv 上，成功率分别达到 **99.3%、89.9%、93.6% 和 81.5%**。
4. **arXiv 摘要（英文节选）：** Vision--Language--Action (VLA) models generate robot actions from visual observations and language instructions, but their understanding of physical dynamics remains limited. Video world models can capture scene dynamics, yet generating future videos introduces high latency into the control loop. We introduce PDRVLA, whose lightweight Physical Dynamics Module extracts physical tokens from vision--language features to condition action generation without online video generation.

**对 wiki 的映射**

- [paper-pdrvla-physical-dynamics-representations](../../wiki/entities/paper-pdrvla-physical-dynamics-representations.md)
