# TuringViT: Making SOTA Vision Transformers Accessible to All（arXiv:2606.24253）

> 来源归档（ingest）

- **标题：** TuringViT: Making SOTA Vision Transformers Accessible to All
- **缩写：** **TuringViT**
- **类型：** paper / vision-transformer / visual-encoder / vlm / efficiency
- **arXiv：** <https://arxiv.org/abs/2606.24253>（PDF：<https://arxiv.org/pdf/2606.24253>）
- **项目页：** <https://turingvit.github.io/>
- **代码：** 截至 2026-07-21 项目页仅列 Technical Report（arXiv），**未列 GitHub / 权重入口**
- **机构：** 小鹏（XPeng）；摘要写明为 XPeng AI 系统统一视觉底座
- **作者（前若干）：** Qiman Wu, Hanlin Chen, Lyujie Chen, Rui Xin, Jianlei Zheng, Mingyuan Wang, Jiahui Hu, Da Zhu 等（完整名单以 arXiv 为准；通讯/顾问含 Hang Zhang, Xianming Liu）
- **入库日期：** 2026-07-21
- **一句话说明：** 面向 VLM/VLA 时代的 **可定制 SOTA ViT**：用 **Turing Linear Attention（TLA）**、**VISTA-Curation** 数据策展与 **原生动态分辨率预训练**，在约 **10% 数据量** 下超过 SigLIP2 等开源骨干，并改善高分辨率延迟缩放。

## 摘录 1：问题与动机

- **痛点：** 下游 VLM/VLA 常直接挂 **SigLIP2** 等现成 ViT，但延迟、时序建模与 VLM 集成需求多样，常需定制 SOTA 级编码器。
- **训练壁垒：** 海量图文数据 + softmax 注意力使高分辨率/动态分辨率预训练代价过高，社区被迫「低分辨率预训练 + 事后适配」。
- **目标：** 在可控算力预算下给出可复现的 **架构–数据–训练** 端到端配方，让 SOTA ViT 可训、可定制、可部署。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-turingvit.md`](../../wiki/entities/paper-turingvit.md)；对照 [Vision Transformer](../../wiki/concepts/vision-transformer.md) 与 [视觉骨干](../../wiki/concepts/vision-backbones.md) 的 **线性注意力 + VLM-native 预训练** 轴。

## 摘录 2：三大设计轴（架构 / 数据 / 训练）

- **Architecture-Efficient — Turing Block：** 每 6 层中 **5× TLA + 1× softmax MHA**；辅以 2D-RoPE、RMSNorm、SwiGLU。序列长度增长时延迟近似平坦。
- **Data-Efficient — VISTA-Curation：** 对噪声网页图文/视频–文本做多候选 recap、相对打分、时序聚合，宣称约 **10× 数据缩减** 不损质量。
- **VLM-Native：** 从一开始就按动态分辨率训；无事后适配/额外参数即可插入下游 VLM；generation-aligned 目标桥接对比式 ViT 与生成式 VLM。
- **四阶段预训练：** (1) MIM 蒸馏 EVA02-CLIP-E（55M，256²）；(2) 有界动态图文 SigLIP+SuperClass（850M）；(3) 无界高分辨率精炼；(4) 图–视频混合（约 2M 视频 + replay）。

**对 wiki 的映射：** 实体页「方法栈 / 流程总览」；与 X-World 系列共享「小鹏统一视觉底座」叙事。

## 摘录 3：结果与开源状态（摘要级）

- **零样本：** TuringViT-24L ImageNet-1K ≈ **83.9**（SigLIP2-L 83.1）；ImageNet-A **89.7**（+5.4）；检索均值优于 SigLIP2-L；预训练样本约 **0.85B** vs SigLIP2-L **10B**。
- **部署叙事：** 摘要称已成为 XPeng AI 系统统一视觉基础；项目页给延迟–分辨率曲线（RTX 3080 Ti / TensorRT FP16）。
- **开源：** 截至入库日 **项目页未列代码/权重** → 按「宣称可复现配方、资产未公开」处理。

**对 wiki 的映射：** 实体页局限与工程实践标明开源边界。
