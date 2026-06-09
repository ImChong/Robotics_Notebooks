# Image Generators are Generalist Vision Learners

> 来源归档（ingest）

- **标题：** Image Generators are Generalist Vision Learners
- **类型：** paper
- **来源：** arXiv abs / HTML；Google DeepMind 出版物页；项目页交叉核对
- **原始链接：**
  - <https://arxiv.org/abs/2604.20329>
  - <https://arxiv.org/pdf/2604.20329>
  - <https://deepmind.google/research/publications/240658/>
  - <https://vision-banana.github.io/>
- **作者：** Valentin Gabeur*, Shangbang Long*, Songyou Peng*（共同一作 / project leads）；Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T. Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin 等；advisors：Yiming Gu, Huizhong Chen；leadership sponsors：Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, Radu Soricut（Google DeepMind）
- **入库日期：** 2026-06-09
- **一句话说明：** 在 **Nano Banana Pro** 图像生成基座上以 **极低比例混入视觉任务数据** 做轻量 **instruction-tuning**，把分割/深度/法线等任务的输出 **参数化为可解码 RGB 图**，在 **不牺牲生成能力** 的前提下于多项 **2D/3D 理解基准** 达到或超越 **SAM 3、Depth Anything 3、Lotus-2** 等专家模型的 **zero-shot transfer** 表现，论证 **生成式视觉预训练** 可作为通用视觉表征学习范式。

## 核心论文摘录（MVP）

### 1) 核心主张：图像生成预训练 ≈ LLM 预训练

- **链接：** <https://arxiv.org/abs/2604.20329>（Abstract / §1）
- **摘录要点：**
  - 近年图像/视频生成模型展现 **零样本视觉理解** 行为，类似 LLM 从生成预训练涌现语言理解。
  - 长期猜想「能创造视觉内容 ⇒ 能理解视觉世界」，但缺乏证据表明 **通用图像生成器** 可在多任务上达到 SOTA 理解能力。
  - 本文论证：**图像生成训练** 类似 **LLM pretraining**，学到强大通用视觉表征；**图像生成** 可作为视觉任务的 **统一接口**（类比文本生成之于 NLP）。
  - 可能标志 CV 范式转移：**生成式视觉预训练** 成为 **Foundation Vision Model** 的中心。
- **对 wiki 的映射：**
  - [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) — 范式定义、与判别/对比学习的对照。
  - [Vision Banana](../../wiki/entities/vision-banana.md) — 实证载体与 benchmark 表。

### 2) 方法：instruction-tuning + RGB 输出空间统一

- **链接：** <https://arxiv.org/abs/2604.20329>（§2 Method）；<https://vision-banana.github.io/>
- **摘录要点：**
  - **基座：** Nano Banana Pro（NBP），预训练图像生成模型。
  - **Vision Banana：** 在 NBP 原训练混合中 **以很低比例** 混入视觉任务数据，做 **轻量 instruction-tuning**（非全量微调、非丢弃生成数据）。
  - **统一接口：** 将视觉任务输出 **参数化为 RGB 图像** → **感知即图像生成**；prompt 指定类别/实例颜色映射或深度 colormap，后处理 **聚类/反演** 回 mask、metric depth、surface normal。
  - **三类优势：**（1）单模型共享权重，仅改 prompt 切换任务；（2）只需少量格式对齐数据；（3）保留原生成先验。
  - **训练数据：** 2D 用 in-house 模型标注的 web 图；3D 用渲染引擎合成；**不含评测基准训练集**。
  - **生成能力保留：** GenAI-Bench 对 NBP **53.5%** win rate；ImgEdit **47.8%**（与基座持平）。
- **对 wiki 的映射：**
  - [Vision Banana](../../wiki/entities/vision-banana.md) — Mermaid 管线、prompt→RGB→解码流程。

### 3) 2D 理解：开放词汇分割 SOTA（zero-shot transfer）

- **链接：** <https://vision-banana.github.io/>（Results — 2D Understanding）
- **摘录要点：**
  - **语义分割** Cityscapes val：**mIoU 69.9**，超 SAM 3（65.2）；开放词汇，prompt 可自然语言/JSON/hex/RGB 指定类–色映射。
  - **实例分割** SA-Co/Gold：**cgF₁ 47.5**（+ Gemini 3.1 Flash-Lite 做 presence），超 OWLv2（24.6）等 zero-shot 方法；每实例动态赋色 + 多阶段聚类解码。
  - **指代表达分割** RefCOCOg UMD：**cIoU 73.8**（超 SAM 3 Agent 73.4）；ReasonSeg：**gIoU 79.3**（+ Gemini 2.5 Pro 将推理 query 转描述性 reference）。
  - 跨任务迁移：未在自由文本 query 上专门训练，仍能理解「墙上图案」「新月形可颂」等细粒度指代。
- **对 wiki 的映射：**
  - [Vision Banana](../../wiki/entities/vision-banana.md) — 2D 任务表与 MLLM 协作模式。
  - [目标检测](../methods/object-detection.md) / [视觉骨干](../../wiki/concepts/vision-backbones.md) — 机器人感知上游对照。

### 4) 3D 理解：单目 metric depth 与 surface normal

- **链接：** <https://vision-banana.github.io/>（Monocular Metric Depth / Surface Normal）
- **摘录要点：**
  - **Metric depth：** 6 数据集平均 **δ₁ 0.929**，超 Depth Anything 3（0.918）；**训练与推理均不使用相机内参**。
  - **Surface normal：** 3 数据集平均角误差 **18.928°**，优于 Lotus-2（19.642°）。
  - 深度图可 **反投影** 为 3D 点云（需数据集内参）；项目页提供交互式点云可视化（NYU / ETH3D 等）。
  - RGB colormap 深度可视化须 **可逆** 回物理深度值方能做定量评测——instruction-tuning 的核心对齐目标之一。
- **对 wiki 的映射：**
  - [Vision Banana](../../wiki/entities/vision-banana.md) — 3D 任务、深度反投影与解码约束。

### 5) 与既有「生成器做理解」路线的差异

- **链接：** <https://arxiv.org/abs/2604.20329>（§1 相关工作段）
- **摘录要点：**
  - 早期工作从生成模型特征抽理解信号，或 zero-shot 生成「像分割/深度图」的 RGB，但 **格式不可控、难定量**。
  - 另一些工作 **加专用模块 + 全量微调** 在单任务达 SOTA，但 **牺牲跨任务泛化与生成能力**。
  - Vision Banana 走 **LLM 式** 路径：**生成基座 + instruction-tuning 格式对齐**，单权重多任务且保留生成。
  - 复杂推理/负样本 presence 等步骤 **外包 MLLM**（Gemini 系列），Vision Banana 专注 **可解码视觉输出**。
- **对 wiki 的映射：**
  - [生成式视觉预训练](../../wiki/concepts/generative-vision-pretraining.md) — 三条技术谱系对比表。
  - [视觉表征作为策略输入](../../wiki/concepts/visual-representation-for-policy.md) — 生成式统一感知作为新兴上游选项。

## 当前提炼状态

- [x] arXiv 摘要、§1–§3 方法与 Table 1 主结果已摘录
- [x] DeepMind 出版物页与 vision-banana.github.io 能力演示、Results 图表已交叉核对
- [x] wiki 映射：`wiki/entities/vision-banana.md`、`wiki/concepts/generative-vision-pretraining.md`
- [ ] 附录细节（实例聚类算法、深度 colormap 反演公式）待按需补强
