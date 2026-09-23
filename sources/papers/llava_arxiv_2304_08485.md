# Visual Instruction Tuning（LLaVA，arXiv:2304.08485）

> 来源归档（ingest）

- **标题：** Visual Instruction Tuning
- **短名：** LLaVA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2304.08485>
- **PDF：** <https://arxiv.org/pdf/2304.08485>
- **项目页：** <https://llava-vl.github.io/>
- **代码：** <https://github.com/haotian-liu/LLaVA>
- **数据：** <https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K>
- **机构：** 威斯康星大学麦迪逊分校（University of Wisconsin–Madison）、微软（Microsoft Research）、哥伦比亚大学（Columbia University）
- **入库日期：** 2026-09-23
- **一句话说明：** 用 GPT-4 生成视觉指令数据，将 CLIP 视觉塔经投影层接到 Vicuna LLM 并两阶段微调；开源 VLM 指令跟随基座，VLA 上游常见「CLIP/LLaVA 式桥接 + 动作头」模板。

## 开源状态（步骤 2.5，2026-09-23）

- **已开源**：`haotian-liu/LLaVA` 训练/推理代码、模型权重与 LLaVA-Instruct-150K 数据；项目页链 Hugging Face 数据集与 demo。

## 核心摘录（面向 wiki 编译）

- **数据**：基于 COCO 图像，用 **language-only GPT-4** 从 caption/bbox 符号表示生成 **158K** 多模态指令样本（对话 / 详细描述 / 复杂推理三类）。
- **模型**：**CLIP ViT-L/14@336px** 视觉编码器 + **线性投影** + **Vicuna** LLM；Stage1 仅训投影（CC3M 对齐），Stage2 端到端指令微调。
- **VLA 读法**：CLIP 解决「看见 ↔ 语言语义对齐」；LLaVA 解决「**多轮视觉指令跟随**」——多数开源 VLA（LlavaVLA、NaVILA 系、RoboInter-VLM 等）复用同一 **冻结视觉塔 + 投影 + LLM** 骨架，再替换/追加动作解码头。
- **后续**：LLaVA-1.5（arXiv:2310.03744）在公开数据上进一步刷榜；本归档以 NeurIPS'23 首版论文为准。
- **对 wiki 的映射：** [paper-llava](../../wiki/entities/paper-llava.md)；模型实体 [llava](../../wiki/entities/llava.md)；上游对齐 [paper-clip](../../wiki/entities/paper-clip.md)

## 当前提炼状态

- [x] 项目页与 GitHub 已交叉核查
- [x] wiki 映射：`wiki/entities/paper-llava.md` 新建
