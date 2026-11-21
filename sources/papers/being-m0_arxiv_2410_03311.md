# Scaling Large Motion Models with Million-Level Human Motions (Being-M0)

> 来源归档（paper / arXiv:2410.03311v3，ICML 2025）

- **URL：** <https://arxiv.org/abs/2410.03311>
- **PDF：** <https://arxiv.org/pdf/2410.03311.pdf>
- **项目页：** <https://beingbeyond.github.io/Being-M0>
- **代码仓库：** <https://github.com/BeingBeyond/Being-M0>
- **标题：** Scaling Large Motion Models with Million-Level Human Motions
- **作者：** Ye Wang, Sipeng Zheng, Bin Cao, Qianshan Wei, Weishuai Zeng, Qin Jin, Zongqing Lu
- **机构：** BeingBeyond（智在无界）、Renmin University of China、CASIA、UCAS、Peking University
- **年份：** 2025（arXiv v1 2024-10；v3 Accepted at ICML 2025）
- **核对范围：** 已获取并通读 `arXiv:2410.03311v3` PDF 抽取全文（Sections 1–6 及 Appendix A–D，涵盖 MotionLib 构建流程、2D-LFQ MotionBook 离散化、LLM 骨干扩展实验及 UNSEEN-90K 评测表）；未独立复现模型训练或推理。
- **通读：** 已完成

## 摘要与核心贡献

1. **MotionLib 百万级数据集**：从 WebVid/Panda-70M 等网络视频经关键词初筛、YOLO-v8 跟踪与切片、关键点置信度过滤（遮挡与低分辨率剔除）、WHAM 3D SMPL 重建 + RL 物理策略 `π_refine`（PHC 式动力学平滑去抖与防滑步）、Gemini-1.5-pro / GPT-4o 分级文本标注（body-level 与 part-level）与规则化语义描述（PoseScript），构建约 **1.21M 片段、1456.4 小时、2.48M 文本标注**的多粒度人类运动语料（规模较既有 HumanML3D/Motion-X 提升 ≥15×）。
2. **MotionBook 与 2D-LFQ 表征**：采用 **SMPL-D135**（9 维根部特征 + 21 关节×6D 连续旋转）替代传统需离线 IK 求解的 HM3D263；将 $T \times D$ 运动序列视作 $T \times D \times 1$ 图像输入 2D CNN，并提出 **2D-LFQ**（Lookup-Free Quantization，$2^{16}=16384$ 隐码），在编码长度缩减为 1/16 的同时显著提升重建保真度与词汇利用率。
3. **LLM Scaling Law（Being-M0）**：系统对比 GPT-2 355M、LLaMA-2 7B/13B、LLaMA-3 8B 及全参数微调 vs LoRA；证明全参数微调显著优于 LoRA，且模型参数与数据规模共扩展时在 **UNSEEN-90K** 域外测试集上 OOD FID 从 HumanML3D 的 `82.674` / Motion-X 的 `70.547` 降至 `11.930`。

## 与知识库关联

- 实体节点：[paper-notebook-scaling-large-motion-models-with-million-level-h.md](../../wiki/entities/paper-notebook-scaling-large-motion-models-with-million-level-h.md)
- 后续工作：[paper-notebook-being-m0-5-a-real-time-controllable-vision-langu.md](../../wiki/entities/paper-notebook-being-m0-5-a-real-time-controllable-vision-langu.md)（HuMo100M / PRQ4）、[paper-notebook-rl-from-physical-feedback-aligning-large-motion.md](../../wiki/entities/paper-notebook-rl-from-physical-feedback-aligning-large-motion.md)（RLPF）
