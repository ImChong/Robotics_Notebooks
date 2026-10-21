# NavAble 数据集页（Hugging Face：NavAble/NeurIPS_2026_BLV）

> 来源归档

- **标题：** NavAble / NeurIPS_2026_BLV
- **类型：** site（数据集页，无独立项目页）
- **URL：** <https://huggingface.co/datasets/NavAble/NeurIPS_2026_BLV>
- **论文：** <https://arxiv.org/abs/2610.19330>（归档 [navable_arxiv_2610_19330](../papers/navable_arxiv_2610_19330.md)）
- **代码：** 论文称管线与 Isaac Sim 扩展发布于同一 HF 地址；页面抓取未显示文件列表（2026-10-17 核查，未核实）
- **数据集：** 同上 URL，cc-by-4.0
- **入库日期：** 2026-10-17
- **一句话说明：** BLV 无障碍导航分割 / 检测数据集，含真实整理、自采与合成三个子集。

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| 元数据 | 图像分割与目标检测；模态 3D / 图像 / 文本；parquet；100K–1M 行 |
| 子集 | real_curated 36.5k；real_ours 5.58k；syn 453k（均为 train split） |
| 字段 | image、mask、source、source_dataset、split_origin、width、height |
| 标签 | accessibility、blind-low-vision、Synthetic、isaacsim、sim-to-real、urban-navigation |

## 开源核查结论

数据集已公开；类别列表、数据卡正文与管线文件在抓取中未显示，不作断言。
