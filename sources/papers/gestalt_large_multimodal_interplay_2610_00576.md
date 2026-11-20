# Gestalt: Large Multimodal Interplay Model｜多模态交互模型

- **类型：** 论文；统一多模态、离散扩散。
- **论文：** https://arxiv.org/html/2610.00576 （arXiv:2610.00576v2，2026-10-08）；PDF：https://arxiv.org/pdf/2610.00576v2
- **机构：** 中国人民大学高瓴人工智能学院、北京大模型与智能治理重点实验室、北京航空航天大学、北京邮电大学、上海人工智能实验室、AresoX（按原文署名）。
- **作者：** Zequn Yang、Yu Miao、Haotian Ni、Ziheng Chen、Chengxiang Huang、Dongzhan Zhou、Kai Chen、Qi Zhang、Ji-Rong Wen、Yake Wei、Di Hu。
- **项目：** https://gewu-lab.github.io/Gestalt/
- **源码：** https://github.com/GeWu-Lab/Gestalt
- **权重：** https://huggingface.co/GeWu-Lab/Gestalt
- **核对日期：** 2026-11-20
- **核对范围：** 通读原文 HTML v2 摘要、引言、§2 方法、§3 实验与表 1–6、§4 相关研究、§5 讨论、§6 数据组织与 §7 附录消融；交叉核对项目页、README 和训练/推理脚本。此归档为原文提炼，非全文转载。
- **通读：** 已完成

## 核心原理

1. **三层 Interplay Pyramid：** 模态独有信息、跨模态冗余对齐、跨模态协同推理。
2. **两段架构：** 初始化 LLaDA-8B-Instruct，使用 IBQ 16,384 视觉码本、MRoPE 和 32 个 learnable interplay tokens。前四层为 Zone I，视觉与文本用独立 FFN，跨模态交换主要经中介 token；后续 Zone II 采用全多模态注意力和共享 FFN。
3. **统一训练：** masked diffusion；70M 联合预训练，8M 条件持续预训练，13.7M 分类 SFT；另有约 2.5M 文生图训练例。SFT 四类为 redundancy、text-unique、visual-unique、synergy。
4. **开源核对：** 有 Stage-III SFT、MMU、T2I 入口和权重；README 仍写 SFT data coming soon。MMU 入口要求预提取图像码本 .npy，T2I 输出 .npy 码本 ID 而不是 PNG。

## 实验与限制

- UniGenBench English Long 综合分 **73.30**（表 1），但 Text 项 **3.80**。
- CV-Bench 2D/3D **78.9 / 86.1**（表 3），为视觉理解指标，不是机器人成功率。
- MM-IMDb / SRBench **67.07 / 53.17**（表 6），是本文多模态协同评测。
- 附录课程消融并非所有指标一致提升；未报告机器人动作头、Sim2Real 或真机闭环结果。
- **同名辨识：** Contra Labs 设计榜的 Gestalt (stealth) 未有证据与本论文同一模型，勿混用身份及权重。

## 对 wiki 的映射

- [Gestalt 论文独立节点](../../wiki/entities/paper-gestalt-2610-00576.md)
- [项目主页](../sites/gestalt-gewu-lab.md) · [官方源码](../repos/gestalt-gewu-lab.md)
