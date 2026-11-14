---
type: entity
tags: [dataset, computer-vision, benchmark, multimodal, multilingual, google]
status: complete
updated: 2026-11-14
arxiv: "2103.01913"
code: https://github.com/google-research-datasets/wit
summary: "Google Research 构建的 108 语种维基百科多模态图文集：包含 3,760 万图文元组、1,150 万唯一图像与页面/章节上下文描述（CC BY-SA 3.0）。"
original:
  - ../../sources/papers/wit_arxiv_2103_01913.md
related:
  - ../entities/transformer-cv-curriculum.md
  - ../concepts/vision-backbones.md
  - ../concepts/multimodality-basics.md
  - ../concepts/generative-vision-pretraining.md
  - ./paper-clip.md
  - ./dataset-flickr30k-entities.md
  - ./dataset-jft-300m.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/papers/wit_arxiv_2103_01913.md
  - ../../sources/courses/transformer_cv_applications_syllabus.md
---

# WIT（Wikipedia-based Image Text）

## 一句话定义

**WIT**（Srinivasan et al., SIGIR 2021 [arXiv:2103.01913](https://arxiv.org/abs/2103.01913)）是 Google Research 从 108 种语言维基百科抽取并经许可与质量过滤的多模态多语言图文数据集，包含 **3,760 万条**带页面与章节语境的多维度图文元组及 **1,150 万张**唯一图像（CC BY-SA 3.0）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|---|---|---|
| WIT | Wikipedia-based Image Text | 本数据集官方全称，面向多模态多语言预训练与实体检索 |
| TFDS | TensorFlow Datasets | 官方提供标准化读取接口的公开数据集分发目录（`tensorflow.org/datasets/catalog/wit`） |
| JSD | Jensen-Shannon Divergence | 论文用来定量衡量 WIT 与 COCO / Flickr30K 词分布差异的对称散度指标 |
| BoW | Bag-of-Words | 论文第 4 节对照实验中使用的轻量级双塔对比学习文本词袋编码基线 |
| CC | Conceptual Captions | Google 早期的百万级英文自动 Alt-Text 图文集，论文中作为域内/域外检索主要对照 |

## 为什么重要

1. **填补真正多语言且可商用开源的大规模图文预训练资源空白：** 早期多模态图文集（COCO、Flickr30K、Conceptual Captions）主要集中在英语且多为日常场景描述；WIT 首次把非英语语种大规模纳入（**108 种语言均 ≥1.2 万元组**，其中 **53 种超过 10 万**，跨语言共享图像的约 **5,000 万+ 跨语种图文对** 覆盖 310 万唯一图像），在开放 CC BY-SA 3.0 协议下支撑 MURAL 等多语言对齐工作。
2. **附带页面、章节与多层图注的结构化语境：** 每条记录不仅给出正文引用图注（Reference，1,720 万条）、Wikimedia 归属说明（Attribution，3,520 万条）和 Alt 文本（540 万条），还完整保留 `page_title`、层级化 `hierarchical_section_title` 与段落描述（1.198 亿条上下文文本），启发后续 WikiWeb2M 等网页多模态理解任务。
3. **揭示「专有实体长尾」与「日常概念描述」之间的根本分布鸿沟：** 论文通过系统性实验指出，强实体、低频专有名词占主导的数据集即使规模十倍于 Conceptual Captions，若直接用于日常物体检索（Flickr30K / COCO）反而表现更低——为选型图文预训练数据敲响「分布匹配优于盲目堆量」的警钟。

## 核心原理

### 数据集速查与 17 列 Schema

- **图像隔离划分：** 严格按唯一图像不重叠切分，**Train 3,713 万行（1,140 万图）**、**Val 26.18 万行（5.8 万图）**、**Test 21.07 万行（5.7 万图）**；总体积约 **~25 GB 压缩 TSV**（10 个训练文件、5 个验证文件、5 个测试文件，外加 1% 抽样测试文件）。
- **17 列核心字段（`DATA.md` 核对）：**
  - 语种与页面定位：`language`、`page_url`、`image_url`、`page_title`、`section_title`、`hierarchical_section_title`、`is_main_image`、`page_changed_recently`
  - 多类图注与上下文：`caption_reference_description`、`caption_attribution_description`、`caption_alt_text_description`、`context_page_description`、`context_section_description`
  - 图像元信息与过滤标志：`mime_type`、`original_height`、`original_width`、`attribution_passes_lang_id`
- **训练输入与具身适配：** 属于静态网页图文语料，可直接用于对比或生成式多模态预训练的图文**训练输入**（CC BY-SA 3.0 开源许可）；不包含三维空间、动作或交互轨迹，面向具身控制须与机器人/动作数据联合**适配**。

### Flume 大规模抽取与多级过滤流水线

```mermaid
flowchart LR
    RAW["维基百科约 1.24 亿页 / 279 语种 (约 1.5 亿原始元组)"] --> TF["文本过滤 (长度>=3 / 剔除.png icon stub 等无效Alt / 限JPG与PNG)"]
    TF --> IF["图像与版权过滤 (>=100x100 px / 仅CC与开放许可 / 剔除占位符图标)"]
    IF --> DS["高频类别重度下采样 (国旗 / 徽标 / 地图 / 徽章)"]
    DS --> SF["安全过滤 (色情与敏感筛选 <0.2% 剔除)"]
    SF --> LF["语种门槛 (>=12K 元组 保留 108 语种)"]
    LF --> SPLIT["按唯一图像严格隔离划分 (Train 37.13M / Val 261.8K / Test 210.7K)"]
```

## 实验与评测

1. **多语言人工质量评测（7 语种约 4,400 对图文、13,000 次人工判定）：** `Reference` 图注相关度（`Yes`+`Maybe`）在英语达 **96.6%**、非英语达 **97.0%**；加上页面/章节上下文后语境一致性在英语达 **99.4%**、非英语达 **98.4%**，全集总体一致性 **98.5%**。
2. **图文检索与域偏移对照（§4 双塔词袋对比学习）：**
   - 组合 `reference + attribution` 文本在检索中效果最优；在 WIT-All 零样本检索中，WIT 训练模型取得 **R@1 = 0.346**（对照 Conceptual Captions 训练模型仅 **0.048**），在多语言维基页面标题预测上同样遥遥领先。
   - **但在 Flickr30K 与 MS-COCO 检索上低于 CC：** 论文深入分析词频与实体分布发现，WIT 中 **72.02%** 的 unigram 频次 ≤ 3（Conceptual Captions 仅为 43%），约 450 万维基百科实体的 **80%+** 出现不超过 3 次，且维基图注习惯直呼地名/人物/建筑物全称而非日常泛化上位词（如不泛称为 `a dog`、`a street`），使其与日常摄影数据集具有更大的 JS 散度。

## 源码运行时序图

**不适用：** 经核对官方仓 [google-research-datasets/wit](https://github.com/google-research-datasets/wit) 全部文件（`README.md`、`DATA.md`、`wikiweb2m.md`、`mural/README.md`、`CONTRIBUTING.md`、`LICENSE`），该仓库仅作为数据集说明文档、17 列 TSV Schema 定义与 GCS/TFDS 下载索引页面存在，**不包含可运行的数据抓取 Flume 管线脚本、本地解码工具或模型训练评测源码**。

## 工程实践

- **优先使用 1% 抽样集与分片 TSV 校验管道：** 完整 10 个训练分片（约 25 GB 文本与 URL 元数据）不包含原始图片像素；调试阶段建议直接拉取官方 1% 抽样包（`wit_v1.train.all-1percent_sample.tsv.gz`）或 Kaggle 竞赛释出的图像像素集（覆盖对应图像超 60%）。
- **合理拼装 `ref` 与 `attr` 上下文提示词：** 在多模态训练中，`caption_reference_description` 描述图像与正文的叙事关联、`caption_attribution_description` 常含拍摄对象与语言标签（受 `attribution_passes_lang_id` 约束）、`context_section_description` 提供背景知识；将 `ref + attr` 并适当引入页面/章节标题可显著稳定对比学习与知识型 VQA。
- **尊重 Wikimedia 抓取速率礼仪与字段限制：** 若根据 `image_url` 从 Wikimedia Commons 批量抓图，务必设置标准 `User-Agent` 并遵守限速，同时留意 `original_height/width`、URL 失效随时间老化以及非英语记录中 `attribution_passes_lang_id` 的语言纯度过滤标志。

## 常见误区与局限

- **把 25 GB 压缩包误认为包含图像像素：** Google 官方仅发布含 `image_url` 的 17 列结构化 TSV（约 25 GB），除 TFDS 部分整合与 Kaggle 比赛子集外，大规模预训练仍需自行合规下载图像或应对部分外链失效。
- **把高语料体量等同于通用日常视觉概念泛化力：** 如实验第 4 节直接证明，未经上位词改写或与日常场景语料（如 CC / LAION / COCO）混采而单独在 WIT 上训练双塔模型，在日常实体与动作描述基准上不占优势。
- **高噪声长尾实体与特定文体偏置：** 维基百科图注包含大量人名、地名、建筑年份、生物拉丁学名及历史黑白文献图片，缺乏室内机器人操作视角、物理动态动作动词与连续时序交互。

## 关联页面

- [Transformer 视觉课程笔记](../entities/transformer-cv-curriculum.md)
- [CLIP](./paper-clip.md)
- [多模态基础概念](../concepts/multimodality-basics.md)
- [生成式视觉预训练](../concepts/generative-vision-pretraining.md)
- [视觉骨干网络](../concepts/vision-backbones.md)
- [Flickr30K Entities](./dataset-flickr30k-entities.md)
- [JFT-300M](./dataset-jft-300m.md)
- [具身评测与基准选择闭环](../queries/embodied-eval-benchmark-selection-loop.md)

## 参考来源

- [WIT 论文与官方仓库归档（arXiv:2103.01913 / SIGIR 2021）](../../sources/papers/wit_arxiv_2103_01913.md) — 本文数据统计、17 列 Schema、Flume 过滤流程、人工与检索实验结论均核对自此一次来源。
- [Transformer CV 应用课程大纲](../../sources/courses/transformer_cv_applications_syllabus.md) — 课程目录初始索引来源。

## 推荐继续阅读

- [CLIP](./paper-clip.md)：理解英文大规模网络图文对比预训练与维基长尾实体图文在泛化特征上的互补关系。
- [Flickr30K Entities](./dataset-flickr30k-entities.md)：对照细粒度短语-区域边界框对齐基准，理解纯图像-整段文本与显式区域定位的差异。
