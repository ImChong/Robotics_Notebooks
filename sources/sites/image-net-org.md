# ImageNet 官方站点（image-net.org）

- **标题：** ImageNet
- **类型：** dataset / research-portal
- **官网：** https://www.image-net.org/
- **竞赛：** https://www.image-net.org/challenges/LSVRC/
- **机构：** Stanford Vision Lab、Stanford University、Princeton University 等（站点页脚与 About）
- **收录日期：** 2026-10-01

## 一句话摘要

ImageNet 项目的 **官方入口**：说明 **WordNet 层次 + synset 级图像** 的目标、**非商用研究下载** 条款、ILSVRC 历史，以及 **CVPR 2009 / IJCV 2015** 等一手出版物 PDF 链接。

## 为何值得保留

- **许可与版权：** 站点明确 **ImageNet 不拥有图像版权**，仅编译 synset 对应的 **网页图像 URL 列表**；下载需注册并遵守条款——产品化与机器人数据集混用前必须法务核对。
- **维护动态：** 含 **2019 人物子树过滤**（FAT* 2020 论文）、**2021 隐私保护更新** 等公告，影响是否仍应默认「全量 ImageNet 预训练权重」。
- **一手 PDF：** About 页链到 [CVPR 2009 ImageNet 论文 PDF](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf) 与 ILSVRC IJCV 论文页面。

## 站点公开要点（编译自 About，2026-10-01 核查）

- **目标：** 为每 synset 平均提供约 **1000** 张 **质控 + 人工标注** 图像，覆盖 WordNet 多数概念。
- **动机：** (1) 建立清晰的 **物体分类 North Star**；(2) 提供 **网络规模训练数据** 以支撑可泛化机器学习。
- **核心团队（PI 级）：** Li Fei-Fei、Jia Deng、Olga Russakovsky、Alex Berg、Kai Li 等（完整贡献者见 About 出版物列表）。

## 代码 / 数据开放核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 数据集下载 | **已开放（受限）**：研究/教育用途经站点注册下载；非 ImageNet 版权方 |
| 官方训练代码 | **无统一官方仓库**；竞赛与社区工具（torchvision、timm、TFDS）为主 |
| 评测服务器 | ILSVRC 历史竞赛 server；近年维护以站点公告为准 |

**判定：** **数据部分开放 + 条款约束**；非「一键复现包」，机器人侧通常使用 **预训练 checkpoint** 或第三方镜像，而非现场拉全库。

## 对 Wiki 的映射

- [`wiki/entities/dataset-imagenet.md`](../../wiki/entities/dataset-imagenet.md)：实体页主来源
- 交叉 [`sources/papers/imagenet_hierarchical_database_cvpr_2009.md`](../papers/imagenet_hierarchical_database_cvpr_2009.md)、[`sources/papers/imagenet_ilsvrc_arxiv_1409_0575.md`](../papers/imagenet_ilsvrc_arxiv_1409_0575.md)
