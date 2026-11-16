# MDF++: Visuo-Tactile World Action Model with End-to-end Representation Learning（arXiv:2611.09293）

> 来源归档（paper）

- **标题：** MDF++: Visuo-Tactile World Action Model with End-to-end Representation Learning
- **类型：** paper
- **作者：** Zixuan Huang, Jinhao Yu, Zachary Charlick, Dmitry Berenson（Department of Robotics, University of Michigan, Ann Arbor）
- **arXiv：** <https://arxiv.org/abs/2611.09293>（v1，2026-11-06，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.09293>
- **项目页：** <https://jointmdf.github.io>（论文摘要给出；未找到公开代码仓库地址）
- **会场：** CoRL 2026（来源：Glen Berseth 的 CoRL 2026 阅读清单，P2，World Models & World Action Models 组；论文首页未写会场）
- **入库日期：** 2026-11-12
- **一句话说明：** 视触觉世界动作模型：在 MDF 扩散骨干上端到端微调观测编码器，用「模态自身去噪损失不回传到自己的编码器」的选择性梯度流避免特征坍缩。
- **核对范围：** 经 arXiv v1 PDF 全文（pdftotext）逐段读完：摘要、1 引言、2 相关工作、3 预备知识、4 方法（4.1 多任务训练、4.2 选择性梯度流）、5 实验（5.1 仿真、5.2 表示分析、5.3 自去噪梯度缩放、5.4 真机）、6 结论、7 局限、附录 A（实现细节与超参 Table 3-6）、附录 B（真机评分标准与 Table 7-9）；图只看了图注和图内文字与数字（Fig. 1-4），参考文献列表只浏览未逐条核对；未运行任何代码，未看项目页视频；OpenReview 页面本环境打不开，未读评审意见。
- **通读：** 已完成

## 核心摘录

1. **问题（引言、2.1）：** 现有 WAM 多依赖冻结的预训练编码器（像素重建或自监督目标），表示未必对控制有用；触觉尚无广泛适用的预训练编码器。作者称直接端到端微调会特征坍缩，即编码器与扩散骨干共适应出「易去噪但对控制弱」的潜变量（4.2）。
2. **方法（4.1-4.2）：** 在 MDF（Multimodal Diffusion Forcing）上，训练时以等概率采样 5 种结构化条件分布（策略、世界动作联合、前向动力学、逆动力学、随机掩码）；对每个观测编码器只回传其他流（含动作）的去噪损失梯度，屏蔽自己那一路（式 2，受 REPA-E 启发）。
3. **架构（4.2、附录 A.1）：** RGB 与触觉图 224x224，分别用 ViT-Tiny/16（ImageNet-21k augreg 初始化）取 CLS token 作潜变量；力、本体、动作用线性层；12 层 Transformer，隐藏维 512，8 头；预测干净 token（x0），余弦噪声表，训练 1000 步，DDIM 10 步推理；200k 步，batch 64；骨干学习率 1e-4，编码器 1e-5；损失权重 动作 4.0、力与本体 1.0、RGB 与触觉 0.01。
4. **仿真（5.1、Table 1）：** 5 任务（Isaac Lab Factory 的 PegInsert、GearMesh、NutThread，加 ManiFeel 的 BallSorting、ObjectSearch），每方法 150 次试验；MDF++ 平均 86.9%，naive grad 64.1%，SIGReg 58.1%，frozen-ViT 58.7%，frozen-Sparsh 58.9%，UWM w/ tactile 74.2%，DP w/ tactile 68.7%，MDF++(policy) 78.9%，MDF(adapted) 68.4%。
5. **表示诊断（5.2、Table 2）：** 在冻结 RGB 特征上训练线性探针预测动作，MSE：MDF++ 0.043，预训练 ViT 0.163，naive E2E 0.520，仅自去噪 0.574；MDF++ 触觉有效秩 33.34，naive E2E 5.99。
6. **真机（5.4、附录 B、Table 9）：** Franka Research 3 + WSG-50，GelSight Mini（只用右指图像），腕部 RealSense；Bolt Screwing 300 条、Nut Mating 255 条 UMI 演示（带 GelSight 的定制 UMI）；每策略每任务 20 次。成功率 MDF++ 80%（16/20）与 40%（8/20），UWM 50% 与 25%，MDF++(frozen-ViT) 30% 与 15%；平均阶段得分 MDF++ 95.0 与 83.0，UWM 57.5 与 61.0，frozen-ViT 46.5 与 19.0。
7. **局限（第 7 节）：** 力信号频率低可能漏掉高频接触动力学；各模态损失权重需手调；选择性梯度流屏蔽自去噪梯度，可能损害潜空间的可扩散性（diffusibility）。

## 与 Glen Berseth CoRL 2026 清单的对照

- 清单一句话：Visuo-tactile world-action model that learns control-relevant representations end-to-end, avoiding feature collapse.
- 要点见 wiki 页「与清单说法的核对」一节。

## 对 wiki 的映射

- [paper-mdf-plus-plus-visuo-tactile-wam](../../wiki/entities/paper-mdf-plus-plus-visuo-tactile-wam.md)
