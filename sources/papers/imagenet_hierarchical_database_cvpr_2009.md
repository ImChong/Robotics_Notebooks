# ImageNet：大规模层次化图像数据库（CVPR 2009）

> 论文来源归档（ingest · 数据集一手资料）

- **标题：** ImageNet: A Large-Scale Hierarchical Image Database
- **作者：** Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei（Princeton University）
- **类型：** paper / dataset / computer-vision / benchmark
- **会议：** IEEE CVPR 2009（Longuet-Higgins Prize 2019 回顾性最具影响力论文）
- **PDF：** <https://image-net.org/static_files/papers/imagenet_cvpr09.pdf>
- **项目页：** <https://www.image-net.org/>
- **入库日期：** 2026-10-01
- **一句话说明：** 以 **WordNet 同义词集（synset）** 为骨架构建 **可公开获取的大规模图像本体**：人工质控、全分辨率、**IS-A 层次**；为 ILSVRC 与后续视觉预训练提供数据组织范式。

## 核心摘录（面向 wiki 编译）

### 1) WordNet 层次与 synset 粒度

- **要点：** 每个 WordNet **synset**（同义词集）对应一个可视觉化的概念；ImageNet 目标是为约 **8 万名词 synset** 各收集 **500–1000** 张干净全分辨率图，形成 **数千万级** 按语义树排序的图像库。
- **对 wiki 的映射：** [`wiki/entities/dataset-imagenet.md`](../../wiki/entities/dataset-imagenet.md)

### 2) 2009 论文报告的规模与质量

- **要点：** 当时版本含 **12 棵子树**（mammal、vehicle、bird 等），**5247 synsets、约 320 万** 图像；随机抽检 **99.7%** 标注精度；相对 Caltech101/256、TinyImages、ESP 等，强调 **LabelDisam（词义消歧）**、**DenseHie（密集层次）**、**FullRes** 与 **PublicAvail**。
- **对 wiki 的映射：** 同上；[`wiki/concepts/vision-backbones.md`](../../wiki/concepts/vision-backbones.md)

### 3) Amazon Mechanical Turk 采集管线

- **要点：** 大规模标注无法靠小团队完成；论文描述 **MTurk 众包 + 质控** 流程，为后续 ILSVRC 百万级标注奠定方法基础。
- **对 wiki 的映射：** [`wiki/entities/dataset-imagenet.md`](../../wiki/entities/dataset-imagenet.md)「核心原理」

### 4) 早期应用验证

- **要点：** 在 **mammal / vehicle** 子树上演示 **目标识别、图像分类、自动聚类**；说明层次化大数据对 **可扩展视觉算法** 的必要性。
- **对 wiki 的映射：** [`wiki/entities/alexnet.md`](../../wiki/entities/alexnet.md)（2012 竞赛突破的数据前提）

## 相关资料索引

| 资料 | 关系 |
|------|------|
| [ILSVRC 综述（IJCV 2015）](imagenet_ilsvrc_arxiv_1409_0575.md) | 基于 ImageNet 子集的标准化竞赛与 1000 类 benchmark |
| [image-net.org 站点归档](../sites/image-net-org.md) | 下载、许可与项目维护入口 |
| [WordNet](https://wordnet.princeton.edu/) | synset 与 IS-A 关系来源 |

## 当前提炼状态

- [x] 要点摘录与 wiki 映射
- [x] 与 ILSVRC / 官方站点交叉链接
