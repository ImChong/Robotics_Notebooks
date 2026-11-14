# HierText：分层场景文本标注数据集

> 官方数据集来源归档（dataset）

- **名称：** HierText
- **机构：** Google Research 数据集维护团队
- **官方仓库：** <https://github.com/google-research-datasets/hiertext>
- **论文：** <https://arxiv.org/abs/2203.15143>（Towards End-to-End Unified Scene Text Detection and Layout Analysis，CVPR 2022）
- **评测源码：** <https://github.com/google-research-datasets/hiertext/blob/main/eval.py>
- **许可：** CC BY-SA 4.0（以官方仓库 LICENSE 为准）
- **入库日期：** 2026-11-14
- **核对范围：** 通读官方 GitHub README 的简介、标注格式、下载、评估命令、许可、引用信息；未实际下载图片或执行评估。
- **通读：** 已完成

## 原始数据及发布状态

- **11,639 张**来自 Open Images 的图像，约 **120 万词级**标注，同时有行、段落层级的多边形信息。
- 划分：train **8,281** / validation **1,724** / test **1,634**。
- 图像通过 CVDF / Open Images 下载，标注文件位于官方仓库 `gt/train.jsonl.gz` 等路径；官方仓库截至 2024-12-02 公告已公开测试集标注。
- 标注结构 `annotations[].paragraphs[].lines[].words[]`；词、行带 `text` / `vertices` / `legible`，行还提供 `handwritten`、`vertical` 属性。
- 官方公布五种任务：词检测、词端到端、行检测、行端到端、段落检测。使用 `eval.py`，发表结果要求 `--mask_stride=1`。
- 该数据集的 **11,639** 张全集不等于 Ultralytics OCR 博客自测使用的 **80 张**验证照片；其 F1 数字不能直接视作 HierText 官方整体榜单结果。

## 开放与复现边界

- **已开放**：标注、评估脚本、可视化 notebook、下载说明；另有论文对应 Unified Detector 模型实现链接。
- **注意**：图片依赖独立的 Open Images 托管位置；仓库已归档只读，不应假定后续维护。
- **许可**：使用、再分发或训练前核对 CC BY-SA 4.0 和所依赖图片的来源条款。

## 对 wiki 的映射

- [HierText 数据集实体](../../wiki/entities/hiertext.md)
- [YOLO26 OCR 方法](../../wiki/methods/yolo26-ocr-document-parsing.md)
